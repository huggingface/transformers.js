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


from argparse import ArgumentParser

from doc_builder.commands.build import build_command_parser
from doc_builder.commands.convert_doc_file import convert_command_parser
from doc_builder.commands.notebook_to_mdx import notebook_to_mdx_command_parser
from doc_builder.commands.preview import preview_command_parser
from doc_builder.commands.push import push_command_parser
from doc_builder.commands.style import style_command_parser


def main():
    parser = ArgumentParser("Doc Builder CLI tool", usage="doc-builder <command> [<args>]")
    subparsers = parser.add_subparsers(help="doc-builder command helpers")

    # Register commands
    convert_command_parser(subparsers=subparsers)
    build_command_parser(subparsers=subparsers)
    notebook_to_mdx_command_parser(subparsers=subparsers)
    style_command_parser(subparsers=subparsers)
    preview_command_parser(subparsers=subparsers)
    push_command_parser(subparsers=subparsers)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    args.func(args)


if __name__ == "__main__":
    main()
