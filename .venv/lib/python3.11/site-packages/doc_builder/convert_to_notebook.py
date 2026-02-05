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

import os
import re
from pathlib import Path

import nbformat

from .autodoc import resolve_links_in_text
from .convert_md_to_mdx import clean_doctest_syntax
from .convert_rst_to_mdx import is_empty_line
from .utils import get_doc_config


# Re pattern that matches inline math in MDX: \\(formula\\)
_re_math_delimiter = re.compile(r"\\\\\((.*?)\\\\\)")
# Re pattern that matches the copyright paragraph in an MDX file
_re_copyright = re.compile(r"<!--\s*Copyright(.*?)-->", flags=re.DOTALL)
# Re pattern that matches YouTube Svelte components and extract the id
_re_youtube = re.compile(r'<Youtube id="([^"]+)"/>')
# Re pattern matching header lines in Markdown
_re_header = re.compile(r"^#+\s+\S+")
# Re pattern matching Python code samples
_re_python_code = re.compile(r"^```\s*(py|python)\s*$")
# Re pattern matching markdown links
_re_markdown_links = re.compile(r"\[([^\]]*)\]\(([^\)]*)\)")
# Re pattern matching framework headers like <pytorch> or <tensorflow>
_re_framework = re.compile(r"^\s*<([a-z]*)>\s*$")


def expand_links(content, page_info):
    """
    Expand links relative to the documentation to full links to the hf.co website.
    """
    package_name = page_info["package_name"]
    version = page_info.get("version", "main")
    language = page_info.get("language", "en")
    page = str(page_info["page"])

    prefix = f"https://huggingface.co/docs/{package_name}/{version}/{language}"

    def _replace_link(match):
        description, link = match.groups()
        if link.startswith("http") or link.startswith("#"):
            return f"[{description}]({link})"
        elif link.startswith("/docs/"):
            return f"[{description}](https://huggingface.co{link})"
        link = "/".join([prefix] + page.split("/")[:-1] + [link])
        return f"[{description}]({link})"

    return _re_markdown_links.sub(_replace_link, content)


def clean_content(content, package=None, mapping=None, page_info=None):
    """
    Clean the content of the doc file to be pure Markdown.
    """
    # Remove copyright
    content = _re_copyright.sub("", content)
    # Remove [[open-in-colab]] marker
    content = content.replace("[[open-in-colab]]\n", "")
    # Replace our special syntax for math formula with the one expected in a notebook.
    content = _re_math_delimiter.sub(r"$\1$", content)
    # Resolve the doc links if possible
    if package is not None and mapping is not None and page_info is not None:
        content = resolve_links_in_text(content, package, mapping, page_info)
        content = expand_links(content, page_info)

    return content.strip()


def split_frameworks(content):
    """
    Split a given doc content in three to extract the Mixed, PyTorch and TensorFlow content.
    """
    new_lines = {"mixed": [], "pt": [], "tf": [], "jax": []}

    content = clean_doctest_syntax(content)
    lines = content.split("\n")
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() == "<frameworkcontent>":
            idx += 1
            current_lines = []
            current_framework = None
            while idx < len(lines) and lines[idx].strip() != "</frameworkcontent>":
                if _re_framework.search(lines[idx]) is not None:
                    current_framework = _re_framework.search(lines[idx]).groups()[0]
                elif current_framework is not None and lines[idx].strip() == f"</{current_framework}>":
                    new_lines[current_framework].extend(current_lines)
                    new_lines["mixed"].extend(current_lines)
                    current_framework = None
                    current_lines = []
                elif current_framework is not None:
                    current_lines.append(lines[idx])
                idx += 1
            idx += 1
        else:
            for key in new_lines.keys():
                new_lines[key].append(lines[idx])
            idx += 1
    return ["\n".join(l) for l in new_lines.values()]


def markdown_cell(content):
    """
    Create a markdown cell with a given content.
    """
    return nbformat.notebooknode.NotebookNode({"cell_type": "markdown", "source": content, "metadata": {}})


def parse_input_output(code_lines):
    """
    Parse a code sample written in doctest syntax to extract input code and expected output.
    """
    current_lines = []
    in_input = True
    cells = []

    for idx, line in enumerate(code_lines):
        if is_empty_line(line):
            current_lines.append(line)
        elif not in_input and line.startswith(">>> "):
            in_input = True
            cells[-1] = (cells[-1][0], "\n".join(current_lines).strip())
            current_lines = [line[4:]]
        elif in_input and line[:4] not in [">>> ", "... "]:
            in_input = False
            cells.append(("\n".join(current_lines).strip(), None))
            current_lines = [line]
        else:
            if line.startswith(">>> ") or line.startswith("... "):
                current_lines.append(line[4:])
            else:
                current_lines.append(line)

    if in_input:
        cells.append(("\n".join(current_lines).strip(), None))
    else:
        cells[-1] = (cells[-1][0], "\n".join(current_lines).strip())

    if len(cells) == 1 and len(cells[0][0]) == 0:
        return [(cells[0][1], None)]

    return cells


def code_cell(code, output=None):
    """
    Create a code cell with some `code` and optionally, `output`.
    """
    if output is None or len(output) == 0:
        outputs = []
    else:
        outputs = [
            nbformat.notebooknode.NotebookNode(
                {
                    "data": {"text/plain": output},
                    "execution_count": None,
                    "metadata": {},
                    "output_type": "execute_result",
                }
            )
        ]
    return nbformat.notebooknode.NotebookNode(
        {"cell_type": "code", "execution_count": None, "source": code, "metadata": {}, "outputs": outputs}
    )


def youtube_cell(youtube_id):
    """
    Create a "YouTube" cell for a given ID. It's actually a code cell with input hidden (requires the hide_input
    extension in regular notebooks and works out of the box on Colab) and the output being the widget with the video.

    Widgets won't be shown by default in a regular notebook unless the user clicks Trust notebook.
    """
    html_code = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{youtube_id}?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>'
    cell_dict = {
        "cell_type": "code",
        "metadata": {"cellView": "form", "hide_input": True},
        "source": ["#@title\n", "from IPython.display import HTML\n", "\n", f"HTML('{html_code}')"],
        "execution_count": None,
    }
    output_dict = {
        "data": {"text/html": [html_code], "text/plain": ["<IPython.core.display.HTML object>"]},
        "execution_count": None,
        "metadata": {},
        "output_type": "execute_result",
    }
    cell_dict["outputs"] = [nbformat.notebooknode.NotebookNode(output_dict)]
    return nbformat.notebooknode.NotebookNode(cell_dict)


def parse_doc_into_cells(content):
    """
    Split a documentation content into cells.
    """
    cells = []
    doc_config = get_doc_config()
    if doc_config is not None and hasattr(doc_config, "notebook_first_cells"):
        for cell in doc_config.notebook_first_cells:
            if cell["type"] == "markdown":
                cells.append(markdown_cell(cell["content"].strip()))
            elif cell["type"] == "code":
                cells.append(code_cell(cell["content"].strip()))

    current_lines = []
    cell_type = "markdown"
    # We keep track of whether we are in a general code block (not necessarily in a code cell) as a line with a comment
    # would be detected as a header.
    in_code = False

    for line in content.split("\n"):
        # Look if we've got a special line.
        special_line = None
        if _re_header.search(line) is not None and not in_code:
            special_line = "header"
        elif _re_python_code.search(line) is not None:
            special_line = "begin_code"
        elif line.rstrip() == "```" and cell_type == "code":
            special_line = "end_code"
        elif line.startswith("```"):
            special_line = "other_code"
        elif _re_youtube.search(line) is not None:
            special_line = "youtube"

        # Some of those special lines mean we have to process the current cell.
        process_current_cell = False
        if cell_type == "markdown":
            process_current_cell = special_line in ["header", "begin_code", "youtube"]
        elif cell_type == "code":
            process_current_cell = special_line == "end_code"

        # Add the current cell to the list
        if process_current_cell:
            if cell_type == "markdown":
                content = "\n".join(current_lines).strip()
                if len(content) > 0:
                    cells.append(markdown_cell(content))
            elif cell_type == "code" and len(current_lines) > 0:
                for code, output in parse_input_output(current_lines):
                    cells.append(code_cell(code, output=output))
            current_lines = []

        if special_line == "header":
            # Header go on their separate Markdown cell, as it plays nicely with the collapsible headers extension.
            cells.append(markdown_cell(line))
        elif special_line == "begin_code":
            cell_type = "code"
            in_code = True
        elif special_line == "end_code":
            cell_type = "markdown"
            in_code = False
        elif special_line == "other_code":
            current_lines.append(line)
            in_code = not in_code
        elif special_line == "youtube":
            # YouTube cells are their own separate cell for proper showing
            youtube_id = _re_youtube.search(line).groups()[0]
            cells.append(youtube_cell(youtube_id))
        else:
            current_lines.append(line)

    # Now that we're done, we just have to process the remainder.
    if cell_type == "markdown":
        content = "\n".join(current_lines).strip()
        if len(content) > 0:
            cells.append(markdown_cell(content))

    return cells


def create_notebook(cells):
    """
    Create a notebook object with `cells`.
    """
    return nbformat.notebooknode.NotebookNode({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 4})


def generate_notebooks_from_file(file_name, output_dir, package=None, mapping=None, page_info=None):
    """
    Generate the notebooks for a given doc file.

    Args:
        file_name (`str` or `os.PathLike`): The doc file to convert.
        output_dir (`str` or `os.PathLike`): Where to save the generated notebooks
        package (`types.ModuleType`, *optional*):
            The package in which to search objects for (needs to be passed to resolve doc links).
        mapping (`Dict[str, str]`, *optional*):
            The map from anchor names of objects to their page in the documentation (needs to be passed to resolve doc
            links).
        page_info (`Dict[str, str]`, *optional*):
            Some information about the page (needs to be passed to resolve doc links).
    """
    output_dirs = [output_dir, os.path.join(output_dir, "pytorch"), os.path.join(output_dir, "tensorflow")]
    output_name = Path(file_name).with_suffix(".ipynb").name
    with open(file_name, "r", encoding="utf-8") as f:
        content = f.read()

    content = clean_content(content, package=package, mapping=mapping, page_info=page_info)

    for folder, content in zip(output_dirs, split_frameworks(content)):
        cells = parse_doc_into_cells(content)
        notebook = create_notebook(cells)
        os.makedirs(folder, exist_ok=True)
        nbformat.write(notebook, os.path.join(folder, output_name), version=4)
