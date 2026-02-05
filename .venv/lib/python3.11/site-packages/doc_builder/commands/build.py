# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import importlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from doc_builder import build_doc, update_versions_file
from doc_builder.utils import (
    get_default_branch_name,
    get_doc_config,
    locate_kit_folder,
    read_doc_config,
    sveltify_file_route,
)


def check_node_is_available():
    try:
        p = subprocess.run(
            ["node", "-v"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
        )
        version = p.stdout.strip()
    except Exception:
        raise EnvironmentError(
            "Using the --html flag requires node v14 to be installed, but it was not found in your system."
        )

    major = int(version[1:].split(".")[0])
    if major < 14:
        raise EnvironmentError(
            "Using the --html flag requires node v14 to be installed, but the version in your system is lower "
            f"({version[1:]})"
        )


def build_command(args):
    read_doc_config(args.path_to_docs)
    if args.html:
        # Error at the beginning if node is not properly installed.
        check_node_is_available()
        # Error at the beginning if we can't locate the kit folder
        kit_folder = locate_kit_folder()
        if kit_folder is None:
            raise EnvironmentError(
                "Using the --html flag requires the kit subfolder of the doc-builder repo. We couldn't find it with "
                "the doc-builder package installed, so you need to run the command from inside the doc-builder repo."
            )

    default_version = get_default_branch_name(args.path_to_docs)
    if args.not_python_module and args.version is None:
        version = default_version
    elif args.version is None:
        module = importlib.import_module(args.library_name)
        version = module.__version__

        if "dev" in version:
            version = default_version
        else:
            version = f"v{version}"
    else:
        version = args.version

    # `version` will always start with prefix `v`
    # `version_tag` does not have to start with prefix `v` (see: https://github.com/huggingface/datasets/tags)
    version_tag = version
    if version != default_version:
        doc_config = get_doc_config()
        version_prefix = getattr(doc_config, "version_prefix", "v")
        version_ = version[1:]  # v2.1.0 -> 2.1.0
        version_tag = f"{version_prefix}{version_}"

    # Disable notebook building for non-master version
    if version != default_version:
        args.notebook_dir = None

    notebook_dir = Path(args.notebook_dir) / args.language if args.notebook_dir is not None else None
    output_path = Path(args.build_dir) / args.library_name / version / args.language

    print("Building docs for", args.library_name, args.path_to_docs, output_path)
    build_doc(
        args.library_name,
        args.path_to_docs,
        output_path,
        clean=args.clean,
        version=version,
        version_tag=version_tag,
        language=args.language,
        notebook_dir=notebook_dir,
        is_python_module=not args.not_python_module,
        version_tag_suffix=args.version_tag_suffix,
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
    )

    # dev build should not update _versions.yml
    package_doc_path = os.path.join(args.build_dir, args.library_name)
    if "pr_" not in version and os.path.isfile(os.path.join(package_doc_path, "_versions.yml")):
        update_versions_file(os.path.join(args.build_dir, args.library_name), version, args.path_to_docs)

    # If asked, convert the MDX files into HTML files.
    if args.html:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            # Copy everything in a tmp dir
            shutil.copytree(kit_folder, tmp_dir / "kit")
            # Manual copy and overwrite from output_path to tmp_dir / "kit" / "src" / "routes"
            # We don't use shutil.copytree as tmp_dir / "kit" / "src" / "routes" exists and contains important files.
            svelte_kit_routes_dir = tmp_dir / "kit" / "src" / "routes"
            for f in output_path.iterdir():
                dest = svelte_kit_routes_dir / f.name
                if f.is_dir():
                    # Remove the dest folder if it exists
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    shutil.copytree(f, dest)
                else:
                    shutil.copy(f, dest)
            # make mdx file paths comply with the sveltekit 1.0 routing mechanism
            # see more: https://learn.svelte.dev/tutorial/pages
            for mdx_file_path in svelte_kit_routes_dir.rglob("*.mdx"):
                new_path = sveltify_file_route(mdx_file_path)
                parent_path = os.path.dirname(new_path)
                os.makedirs(parent_path, exist_ok=True)
                shutil.move(mdx_file_path, new_path)

            # Move the objects.inv file at the root
            if not args.not_python_module:
                shutil.move(tmp_dir / "kit" / "src" / "routes" / "objects.inv", tmp_dir / "objects.inv")

            # Build doc with node
            working_dir = str(tmp_dir / "kit")
            print("Installing node dependencies")
            subprocess.run(
                ["npm", "ci"],
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=working_dir,
            )

            env = os.environ.copy()
            env["DOCS_LIBRARY"] = (
                env["package_name"] or args.library_name if "package_name" in env else args.library_name
            )
            env["DOCS_VERSION"] = version
            env["DOCS_LANGUAGE"] = args.language
            print("Building HTML files. This will take a while :-)")
            subprocess.run(
                ["npm", "run", "build"],
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8",
                cwd=working_dir,
                env=env,
            )

            # Copy result back in the build_dir.
            shutil.rmtree(output_path)
            shutil.copytree(tmp_dir / "kit" / "build", output_path)
            # Move the objects.inv file back
            if not args.not_python_module:
                shutil.move(tmp_dir / "objects.inv", output_path / "objects.inv")


def build_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("build")
    else:
        parser = argparse.ArgumentParser("Doc Builder build command")

    parser.add_argument("library_name", type=str, help="Library name")
    parser.add_argument(
        "path_to_docs",
        type=str,
        help="Local path to library documentation. The library should be cloned, and the folder containing the "
        "documentation files should be indicated here.",
    )
    parser.add_argument("--build_dir", type=str, help="Where the built documentation will be.", default="./build/")
    parser.add_argument("--clean", action="store_true", help="Whether or not to clean the output dir before building.")
    parser.add_argument("--language", type=str, help="Language of the documentation to generate", default="en")
    parser.add_argument(
        "--version",
        type=str,
        help="Version of the documentation to generate. Will default to the version of the package module (using "
        "`main` for a version containing dev).",
    )
    parser.add_argument("--notebook_dir", type=str, help="Where to save the generated notebooks.", default=None)
    parser.add_argument("--html", action="store_true", help="Whether or not to build HTML files instead of MDX files.")
    parser.add_argument(
        "--not_python_module",
        action="store_true",
        help="Whether docs files do NOT have corresponding python module (like HF course & hub docs).",
    )
    parser.add_argument(
        "--version_tag_suffix",
        type=str,
        default="src/",
        help="Suffix to add after the version tag (e.g. 1.3.0 or main) in the documentation links. For example, the default `src/` suffix will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/src/`.",
    )
    parser.add_argument(
        "--repo_owner",
        type=str,
        default="huggingface",
        help="Owner of the repo (e.g. huggingface, rwightman, etc.).",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help="Name of the repo (e.g. transformers, pytorch-image-models, etc.). By default, this is the same as the library_name.",
    )
    if subparsers is not None:
        parser.set_defaults(func=build_command)
    return parser
