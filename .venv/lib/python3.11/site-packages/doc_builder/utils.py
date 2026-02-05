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

import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import yaml
from packaging import version as package_version


hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "doc_builder")
DOC_BUILDER_CACHE = os.getenv("DOC_BUILDER_CACHE", default_cache_path)


def get_default_branch_name(repo_folder):
    config = get_doc_config()
    if config is not None and hasattr(config, "default_branch_name"):
        print(config.default_branch_name)
        return config.default_branch_name
    try:
        p = subprocess.run(
            "git symbolic-ref refs/remotes/origin/HEAD".split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=repo_folder,
        )
        branch = p.stdout.strip().split("/")[-1]
        return branch
    except Exception:
        # Just in case git is not installed, we need a default
        return "main"


def update_versions_file(build_path, version, doc_folder):
    """
    Insert new version into _versions.yml file of the library
    Assumes that _versions.yml exists and has its first entry as main version
    """
    main_branch = get_default_branch_name(doc_folder)
    if version == main_branch:
        return
    with open(os.path.join(build_path, "_versions.yml"), "r") as versions_file:
        versions = yaml.load(versions_file, yaml.FullLoader)

        if versions[0]["version"] != main_branch:
            raise ValueError(f"{build_path}/_versions.yml does not contain a {main_branch} version")

        main_version, sem_versions = versions[0], versions[1:]
        new_version = {"version": version}
        did_insert = False
        for i, value in enumerate(sem_versions):
            if package_version.parse(new_version["version"]) == package_version.parse(value["version"]):
                # Nothing to do, the version is here already.
                return
            elif package_version.parse(new_version["version"]) > package_version.parse(value["version"]):
                sem_versions.insert(i, new_version)
                did_insert = True
                break
        if not did_insert:
            sem_versions.append(new_version)

    with open(os.path.join(build_path, "_versions.yml"), "w") as versions_file:
        versions_updated = [main_version] + sem_versions
        yaml.dump(versions_updated, versions_file)


doc_config = None


def read_doc_config(doc_folder):
    """
    Execute the `_config.py` file inside the doc source directory and executes it as a Python module.
    """
    global doc_config

    if os.path.isfile(os.path.join(doc_folder, "_config.py")):
        loader = importlib.machinery.SourceFileLoader("doc_config", os.path.join(doc_folder, "_config.py"))
        spec = importlib.util.spec_from_loader("doc_config", loader)
        doc_config = importlib.util.module_from_spec(spec)
        loader.exec_module(doc_config)


def get_doc_config():
    """
    Returns the `doc_config` if it has been loaded.
    """
    return doc_config


def is_watchdog_available():
    """
    Checks if soft dependency `watchdog` exists.
    """
    return importlib.util.find_spec("watchdog") is not None


def is_doc_builder_repo(path):
    """
    Detects whether a folder is the `doc_builder` or not.
    """
    setup_file = Path(path) / "setup.py"
    if not setup_file.exists():
        return False
    with open(os.path.join(path, "setup.py")) as f:
        first_line = f.readline()
    return first_line == "# Doc-builder package setup.\n"


def locate_kit_folder():
    """
    Returns the location of the `kit` folder of `doc-builder`.

    Will clone the doc-builder repo and cache it, if it's not found.
    """
    # First try: let's search where the module is.
    repo_root = Path(__file__).parent.parent.parent
    kit_folder = repo_root / "kit"
    if kit_folder.is_dir():
        return kit_folder

    # Second try, maybe we are inside the doc-builder repo
    current_dir = Path.cwd()
    while current_dir.parent != current_dir and not (current_dir / ".git").is_dir():
        current_dir = current_dir.parent
    kit_folder = current_dir / "kit"
    if kit_folder.is_dir() and is_doc_builder_repo(current_dir):
        return kit_folder

    # Otherwise, let's clone the repo and cache it.
    return Path(get_cached_repo()) / "kit"


def get_cached_repo():
    """
    Clone and cache the `doc-builder` repo.
    """
    os.makedirs(DOC_BUILDER_CACHE, exist_ok=True)
    cache_repo_path = Path(DOC_BUILDER_CACHE) / "doc-builder-repo"
    if not cache_repo_path.is_dir():
        print(
            "To build the HTML doc, we need the kit subfolder of the `doc-builder` repo. Cloning it and caching at "
            f"{cache_repo_path}."
        )
        _ = subprocess.run(
            "git clone https://github.com/huggingface/doc-builder.git".split(),
            stderr=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=DOC_BUILDER_CACHE,
        )
        shutil.move(Path(DOC_BUILDER_CACHE) / "doc-builder", cache_repo_path)
    else:
        _ = subprocess.run(
            ["git", "pull"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=cache_repo_path,
        )
    return cache_repo_path


def sveltify_file_route(filename):
    """
    Given `filename` /path/abc/xyz.mdx, return /path/abc/xyz/+page.svelte
    """
    # filename can be PosixPath or str
    filename = str(filename)
    # Check if the filename ends with '.svelte'
    if filename.endswith(".mdx"):
        # Replace the '{name}.mdx' with '{name}/+page.svelte'
        return filename.rsplit(".", 1)[0] + "/+page.svelte"
    return filename
