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
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Thread

from doc_builder import build_doc
from doc_builder.commands.build import check_node_is_available, locate_kit_folder
from doc_builder.commands.convert_doc_file import find_root_git
from doc_builder.utils import is_watchdog_available, read_doc_config


if is_watchdog_available():
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class WatchEventHandler(FileSystemEventHandler):
        """
        Utility class for building updated mdx files when a file change event is recorded.
        """

        def __init__(self, args, source_files_mapping, kit_routes_folder):
            super().__init__()
            self.args = args
            self.source_files_mapping = source_files_mapping
            self.kit_routes_folder = kit_routes_folder

        def on_created(self, event):
            super().on_created(event)
            is_valid, src_path, relative_path = self.transform_path(event)
            if is_valid:
                self.build(src_path, relative_path)

        def on_modified(self, event):
            super().on_modified(event)
            is_valid, src_path, relative_path = self.transform_path(event)
            if is_valid:
                self.build(src_path, relative_path)

        def transform_path(self, event):
            """
            Check if a file is a doc file (mdx, or py file used as autodoc).
            If so, returns mdx file path.
            """
            src_path = event.src_path
            parent_path_absolute = str(Path(self.args.path_to_docs).absolute())
            relative_path = event.src_path[len(parent_path_absolute) + 1 :]
            is_valid_file = False
            if not event.is_directory:
                if src_path.endswith(".py") and src_path in self.source_files_mapping:
                    src_path = self.source_files_mapping[src_path]
                # if src_path.endswith(".md"):
                #     # src_path += "x"
                #     relative_path += "x"
                if src_path.endswith(".mdx") or src_path.endswith(".md"):
                    is_valid_file = True
                    return is_valid_file, src_path, relative_path
            return is_valid_file, src_path, relative_path

        def build(self, src_path, relative_path):
            """
            Build single mdx file in a temp dir.
            """
            print(f"Building: {src_path}")
            try:
                # copy the built files into the actual build folder dawg
                with tempfile.TemporaryDirectory() as tmp_input_dir:
                    # copy the file into tmp_input_dir
                    shutil.copy(src_path, tmp_input_dir)

                    with tempfile.TemporaryDirectory() as tmp_out_dir:
                        build_doc(
                            self.args.library_name,
                            tmp_input_dir,
                            tmp_out_dir,
                            version=self.args.version,
                            language=self.args.language,
                            is_python_module=not self.args.not_python_module,
                            watch_mode=True,
                        )
                        if str(src_path).endswith(".md"):
                            src_path += "x"
                            relative_path += "x"
                        src = Path(tmp_out_dir) / Path(src_path).name
                        dest = self.kit_routes_folder / relative_path
                        shutil.move(src, dest)
            except Exception as e:
                print(f"Error building: {src_path}\n{e}")


def start_watcher(path, event_handler):
    """
    Starts `pywatchdog.observer` for listening changes in `path`.
    """
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print(f"\nWatching for changes in: {path}\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def start_sveltekit_dev(tmp_dir, env, args):
    """
    Installs sveltekit node dependencies & starts sveltekit in dev mode in a temp dir.
    """
    working_dir = str(tmp_dir / "kit")
    print("Installing node dependencies")
    subprocess.run(
        ["npm", "ci"],
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
        cwd=working_dir,
        shell=platform.system() == "Windows",
    )

    # start sveltekit in dev mode
    subprocess.run(
        ["npm", "run", "dev"],
        check=True,
        encoding="utf-8",
        cwd=working_dir,
        env=env,
        shell=platform.system() == "Windows",
    )


def preview_command(args):
    if not is_watchdog_available():
        raise ImportError(
            "Please install `watchdog` to run `doc-builder preview` command.\nYou can do so through pip: `pip install watchdog`"
        )

    read_doc_config(args.path_to_docs)
    # Error at the beginning if node is not properly installed.
    check_node_is_available()
    # Error at the beginning if we can't locate the kit folder
    kit_folder = locate_kit_folder()
    if kit_folder is None:
        raise EnvironmentError(
            "Requires the kit subfolder of the doc-builder repo. We couldn't find it with "
            "the doc-builder package installed, so you need to run the command from inside the doc-builder repo."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / args.library_name / args.version / args.language

        print("Initial build docs for", args.library_name, args.path_to_docs, output_path)
        source_files_mapping = build_doc(
            args.library_name,
            args.path_to_docs,
            output_path,
            clean=True,
            version=args.version,
            language=args.language,
            is_python_module=not args.not_python_module,
        )

        # convert the MDX files into HTML files.
        tmp_dir = Path(tmp_dir)
        # Copy everything in a tmp dir
        shutil.copytree(kit_folder, tmp_dir / "kit")
        # Manual copy and overwrite from output_path to tmp_dir / "kit" / "src" / "routes"
        # We don't use shutil.copytree as tmp_dir / "kit" / "src" / "routes" exists and contains important files.
        kit_routes_folder = tmp_dir / "kit" / "src" / "routes"
        # files/folders cannot have a name that starts with `__` since it is a reserved Sveltekit keyword
        for p in output_path.glob("**/*__*"):
            if p.exists():
                p.rmdir if p.is_dir() else p.unlink()
        for f in output_path.iterdir():
            dest = kit_routes_folder / f.name
            if f.is_dir():
                # Remove the dest folder if it exists
                if dest.is_dir():
                    shutil.rmtree(dest)
                shutil.copytree(f, dest)
            else:
                shutil.copy(f, dest)

        # Node
        env = os.environ.copy()
        env["DOCS_LIBRARY"] = env["package_name"] or args.library_name if "package_name" in env else args.library_name
        env["DOCS_VERSION"] = args.version
        env["DOCS_LANGUAGE"] = args.language
        Thread(target=start_sveltekit_dev, args=(tmp_dir, env, args)).start()

        git_folder = find_root_git(args.path_to_docs)
        event_handler = WatchEventHandler(args, source_files_mapping, kit_routes_folder)
        start_watcher(git_folder, event_handler)


def preview_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("preview")
    else:
        parser = argparse.ArgumentParser("Doc Builder preview command")

    parser.add_argument("library_name", type=str, help="Library name")
    parser.add_argument(
        "path_to_docs",
        type=str,
        help="Local path to library documentation. The library should be cloned, and the folder containing the "
        "documentation files should be indicated here.",
    )
    parser.add_argument("--language", type=str, help="Language of the documentation to generate", default="en")
    parser.add_argument("--version", type=str, help="Version of the documentation to generate", default="main")
    parser.add_argument(
        "--not_python_module",
        action="store_true",
        help="Whether docs files do NOT have corresponding python module (like HF course & hub docs).",
    )

    if subparsers is not None:
        parser.set_defaults(func=preview_command)
    return parser
