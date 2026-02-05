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
import re
from pathlib import Path

from doc_builder.autodoc import is_rst_docstring, remove_example_tags
from doc_builder.convert_rst_to_mdx import (
    apply_min_indent,
    base_rst_to_mdx,
    convert_rst_to_mdx,
    find_indent,
    is_empty_line,
)


def find_docstring_indent(docstring):
    """
    Finds the indent in the first nonempty line.
    """
    for line in docstring.split("\n"):
        if not is_empty_line(line):
            return find_indent(line)


def find_root_git(folder):
    "Finds the first parent folder who is a git directory or returns None if there is no git directory."
    folder = Path(folder).absolute()
    while folder != folder.parent and not (folder / ".git").exists():
        folder = folder.parent
    return folder if folder != folder.parent else None


# Re pattern that matches links of the form [`some_class`]
_re_internal_ref = re.compile(r"\[`([^`]*)`\]")


def shorten_internal_refs(content):
    """
    Shortens links of the form [`~transformers.Trainer`] to just [`Trainer`].
    """

    def _shorten_ref(match):
        full_name = match.groups()[0]
        full_name = full_name.replace("transformers.", "")
        if full_name.startswith("~") and "." not in full_name:
            full_name = full_name[1:]
        return f"[`{full_name}`]"

    return _re_internal_ref.sub(_shorten_ref, content)


def convert_rst_file(source_file, output_file, page_info):
    with open(source_file, "r", encoding="utf-8") as f:
        text = f.read()

    text = convert_rst_to_mdx(text, page_info, add_imports=False)
    text = text.replace("&amp;lcub;", "{")
    text = text.replace("&amp;lt;", "<")
    text = re.sub(r"^\[\[autodoc\]\](\s+)(transformers\.)", r"[[autodoc]]\1", text, flags=re.MULTILINE)
    text = shorten_internal_refs(text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


def convert_rst_docstring_to_markdown(docstring, page_info):
    """
    Convert a given docstring in rst format to Markdown.
    """
    min_indent = find_docstring_indent(docstring)
    docstring = base_rst_to_mdx(docstring, page_info, unindent=False)
    docstring = remove_example_tags(docstring)
    docstring = shorten_internal_refs(docstring)
    docstring = apply_min_indent(docstring, min_indent)
    docstring = docstring.replace("&amp;lcub;", "{")
    docstring = docstring.replace("&amp;lt;", "<")
    return docstring


def convert_rst_docstrings_in_file(source_file, output_file, page_info):
    with open(source_file, "r", encoding="utf-8") as f:
        code = f.read()
    docstrings = code.split('"""')

    for idx, docstring in enumerate(docstrings):
        if idx % 2 == 0 or not is_rst_docstring(docstring):
            continue
        docstrings[idx] = convert_rst_docstring_to_markdown(docstring, page_info)

    code = '"""'.join(docstrings)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(code)


def convert_command(args):
    source_file = Path(args.source_file).absolute()
    if source_file.suffix not in [".rst", ".py"]:
        raise ValueError(f"This script only converts rst files. Got {source_file}.")
    if args.package_name is None:
        git_folder = find_root_git(source_file)
        if git_folder is None:
            raise ValueError(
                "Cannot determine a default for package_name as the file passed is not in a git directory. "
                "Please pass along a package_name."
            )
        package_name = git_folder.name
    else:
        package_name = args.package_name

    if args.output_file is None:
        output_file = source_file.with_suffix(".mdx") if source_file.suffix == ".rst" else source_file
    else:
        output_file = args.output_file

    page_info = {"package_name": package_name, "no_prefix": True}

    if source_file.suffix == ".py":
        convert_rst_docstrings_in_file(source_file, output_file, page_info)

    else:
        if args.doc_folder is None:
            git_folder = find_root_git(source_file)
            if git_folder is None:
                raise ValueError(
                    "Cannot determine a default for package_name as the file passed is not in a git directory. "
                    "Please pass along a package_name."
                )
            doc_folder = (git_folder / "docs") / "source"
            if doc_folder / source_file.relative_to(doc_folder) != source_file:
                raise ValueError(
                    f"The default found for `doc_folder` is {doc_folder} but it does not look like {source_file} is "
                    "inside it."
                )
        else:
            doc_folder = args.doc_folder

        page_info["page"] = source_file.with_suffix(".html").relative_to(doc_folder)

        convert_rst_file(source_file, output_file, page_info)


def convert_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("convert")
    else:
        parser = argparse.ArgumentParser("Doc Builder convert command")

    parser.add_argument("source_file", type=str, help="The file to convert.")
    parser.add_argument(
        "--package_name",
        type=str,
        default=None,
        help="The name of the package this doc file belongs to. Will default to the name of the root git repo "
        "`source_file` is in.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Where to save the converted file. Will default to the `source_file` with an mdx suffix for rst files,"
        "`source_file` for a py file.",
    )
    parser.add_argument(
        "--doc_folder",
        type=str,
        help="The path to the folder with the doc source files. Will default to the `docs/source` subfolder of the "
        "root git repo.",
    )

    if subparsers is not None:
        parser.set_defaults(func=convert_command)
    return parser
