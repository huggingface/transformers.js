# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import logging
import shutil
import subprocess
from pathlib import Path
from time import sleep, time

from huggingface_hub import HfApi


REPO_TYPE = "dataset"
SEPARATOR = "/"


def create_zip_name(library_name, version, with_ext=True):
    file_name = f"{library_name}{SEPARATOR}{version}"
    if with_ext:
        file_name += ".zip"
    return file_name


def push_command(args):
    """
    Commit file doc builds changes using: 1. zip doc build artifacts 2. hf_hub client to upload/delete zip file
    Usage: doc-builder push $args
    """
    if args.n_retries < 1:
        raise ValueError(f"CLI arg `n_retries` MUST be positive & non-zero; supplied value was {args.n_retries}")
    if args.is_remove:
        push_command_remove(args)
    else:
        push_command_add(args)


def push_command_add(args):
    """
    Commit file changes using: 1. zip doc build artifacts 2. hf_hub client to upload zip file
    Used in: build_main_documentation.yml & build_pr_documentation.yml
    """
    max_n_retries = args.n_retries + 1
    number_of_retries = args.n_retries
    n_seconds_sleep = 5

    library_name = args.library_name
    path_docs_built = Path(library_name)
    doc_version_folder = next(filter(lambda x: not x.is_file(), path_docs_built.glob("*")), None).relative_to(
        path_docs_built
    )
    doc_version_folder = str(doc_version_folder)

    zip_file_path = create_zip_name(library_name, doc_version_folder)
    # eg create ./transformers/v4.0.zip with '/transformers/v4.0/*' file architecture inside
    # Use subprocess.run instead of shutil.make_archive to avoid corrupted files, see https://github.com/huggingface/doc-builder/issues/348
    print(f"Running zip command: zip -r {zip_file_path} {path_docs_built}")
    subprocess.run(["zip", "-r", zip_file_path, path_docs_built], check=True)

    api = HfApi()

    time_start = time()

    while number_of_retries:
        try:
            if args.upload_version_yml:
                # removing doc artifact folder to upload 2 files using `upload_folder`: _version.yml and zipped doc artifact file
                shutil.rmtree(f"{library_name}/{doc_version_folder}")
                api.upload_folder(
                    repo_id=args.doc_build_repo_id,
                    repo_type=REPO_TYPE,
                    folder_path=library_name,
                    path_in_repo=library_name,
                    commit_message=args.commit_msg,
                    token=args.token,
                )
            else:
                api.upload_file(
                    repo_id=args.doc_build_repo_id,
                    repo_type=REPO_TYPE,
                    path_or_fileobj=zip_file_path,
                    path_in_repo=zip_file_path,
                    commit_message=args.commit_msg,
                    token=args.token,
                )
            break
        except Exception as e:
            number_of_retries -= 1
            print(f"push_command_add error occurred: {e}")
            if number_of_retries:
                print(f"Failed on try #{max_n_retries-number_of_retries}, pushing again in {n_seconds_sleep} seconds")
                sleep(n_seconds_sleep)
            else:
                raise RuntimeError("push_command_add failed") from e

    time_end = time()
    logging.debug(f"push_command_add took {time_end-time_start:.4f} seconds or {(time_end-time_start)/60.0:.2f} mins")


def push_command_remove(args):
    """
    Commit file deletions using hf_hub client to delete zip file
    Used in: delete_doc_comment.yml
    """
    max_n_retries = args.n_retries + 1
    number_of_retries = args.n_retries
    n_seconds_sleep = 5

    library_name = args.library_name
    doc_version_folder = args.doc_version
    doc_build_repo_id = args.doc_build_repo_id
    commit_msg = args.commit_msg

    api = HfApi()
    zip_file_path = create_zip_name(library_name, doc_version_folder)

    while number_of_retries:
        try:
            api.delete_file(
                zip_file_path, doc_build_repo_id, token=args.token, repo_type=REPO_TYPE, commit_message=commit_msg
            )
            break
        except Exception as e:
            number_of_retries -= 1
            print(f"push_command_remove error occurred: {e}")
            if number_of_retries:
                print(f"Failed on try #{max_n_retries-number_of_retries}, pushing again in {n_seconds_sleep} seconds")
                sleep(n_seconds_sleep)
            else:
                raise RuntimeError("push_command_remove failed") from e


def push_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("push")
    else:
        parser = argparse.ArgumentParser("Doc Builder push command")

    parser.add_argument(
        "library_name",
        type=str,
        help="The name of the library, which also acts as a path where built doc artifacts reside in",
    )
    parser.add_argument(
        "--doc_build_repo_id",
        type=str,
        help="Repo to which doc artifacts will be committed (e.g. `huggingface/doc-build-dev`)",
    )
    parser.add_argument("--token", type=str, help="Github token that has write/push permission to `doc_build_repo_id`")
    parser.add_argument(
        "--commit_msg",
        type=str,
        help="Git commit message",
        default="Github GraphQL createcommitonbranch commit",
    )
    parser.add_argument("--n_retries", type=int, help="Number of push retries in the event of conflict", default=1)
    parser.add_argument(
        "--doc_version",
        type=str,
        default=None,
        help="Version of the generated documentation.",
    )
    parser.add_argument(
        "--is_remove",
        action="store_true",
        help="Whether or not to remove entire folder ('--doc_version') from git tree",
    )
    parser.add_argument(
        "--upload_version_yml",
        action="store_true",
        help="Whether or not to push _version.yml file to git repo",
    )

    if subparsers is not None:
        parser.set_defaults(func=push_command)
    return parser
