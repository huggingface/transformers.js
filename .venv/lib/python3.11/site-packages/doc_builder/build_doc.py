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


import importlib
import os
import re
import shutil
import zlib
from pathlib import Path

import yaml
from tqdm import tqdm

from .autodoc import autodoc, find_object_in_package, get_source_path, resolve_links_in_text
from .convert_md_to_mdx import convert_md_to_mdx
from .convert_rst_to_mdx import convert_rst_to_mdx, find_indent, is_empty_line
from .convert_to_notebook import generate_notebooks_from_file
from .utils import get_doc_config, read_doc_config


_re_autodoc = re.compile(r"^\s*\[\[autodoc\]\]\s+(\S+)\s*$")
_re_list_item = re.compile(r"^\s*-\s+(\S+)\s*$")


def resolve_open_in_colab(content, page_info):
    """
    Replaces [[open-in-colab]] special markers by the proper svelte component.

    Args:
        content (`str`): The documentation to treat.
        page_info (`Dict[str, str]`, *optional*): Some information about the page.
    """
    if "[[open-in-colab]]" not in content:
        return content

    package_name = page_info["package_name"]
    language = page_info.get("language", "en")
    page_name = Path(page_info["page"]).stem
    nb_prefix = f"/github/huggingface/notebooks/blob/main/{package_name}_doc/{language}/"
    nb_prefix_colab = f"https://colab.research.google.com{nb_prefix}"
    nb_prefix_awsstudio = f"https://studiolab.sagemaker.aws/import{nb_prefix}"
    links = [
        ("Mixed", f"{nb_prefix_colab}{page_name}.ipynb"),
        ("PyTorch", f"{nb_prefix_colab}pytorch/{page_name}.ipynb"),
        ("TensorFlow", f"{nb_prefix_colab}tensorflow/{page_name}.ipynb"),
        ("Mixed", f"{nb_prefix_awsstudio}{page_name}.ipynb"),
        ("PyTorch", f"{nb_prefix_awsstudio}pytorch/{page_name}.ipynb"),
        ("TensorFlow", f"{nb_prefix_awsstudio}tensorflow/{page_name}.ipynb"),
    ]
    formatted_links = ['    {label: "' + key + '", value: "' + value + '"},' for key, value in links]

    svelte_component = """<DocNotebookDropdown
  classNames="absolute z-10 right-0 top-0"
  options={[
"""
    svelte_component += "\n".join(formatted_links)
    svelte_component += "\n]} />"

    return content.replace("[[open-in-colab]]", svelte_component)


def resolve_autodoc(content, package, return_anchors=False, page_info=None, version_tag_suffix="src/"):
    """
    Replaces [[autodoc]] special syntax by the corresponding generated documentation in some content.

    Args:
        content (`str`): The documentation to treat.
        package (`types.ModuleType`): The package where to look for objects to document.
        return_anchors (`bool`, *optional*, defaults to `False`):
            Whether or not to return the list of anchors generated.
        page_info (`Dict[str, str]`, *optional*): Some information about the page.
        version_tag_suffix (`str`, *optional*, defaults to `"src/"`):
            Suffix to add after the version tag (e.g. 1.3.0 or main) in the documentation links.
            For example, the default `"src/"` suffix will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/src/`.
            For example, `version_tag_suffix=""` will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/`.
    """
    idx_last_heading = None
    is_inside_codeblock = False
    lines = content.split("\n")
    new_lines = []
    source_files = None
    if return_anchors:
        anchors = []
        errors = []
    idx = 0
    while idx < len(lines):
        if _re_autodoc.search(lines[idx]) is not None:
            object_name = _re_autodoc.search(lines[idx]).groups()[0]
            autodoc_indent = find_indent(lines[idx])
            idx += 1
            while idx < len(lines) and is_empty_line(lines[idx]):
                idx += 1
            if (
                idx < len(lines)
                and find_indent(lines[idx]) > autodoc_indent
                and _re_list_item.search(lines[idx]) is not None
            ):
                methods = []
                methods_indent = find_indent(lines[idx])
                while is_empty_line(lines[idx]) or (
                    find_indent(lines[idx]) == methods_indent and _re_list_item.search(lines[idx]) is not None
                ):
                    if not is_empty_line(lines[idx]):
                        methods.append(_re_list_item.search(lines[idx]).groups()[0])
                    idx += 1
                    if idx >= len(lines):
                        break
            else:
                methods = None
            doc = autodoc(
                object_name,
                package,
                methods=methods,
                return_anchors=return_anchors,
                page_info=page_info,
                version_tag_suffix=version_tag_suffix,
            )
            if return_anchors:
                if len(doc[1]) and idx_last_heading is not None:
                    object_anchor = doc[1][0]
                    new_lines[idx_last_heading] += f"[[{object_anchor}]]"
                    idx_last_heading = None
                anchors.extend(doc[1])
                errors.extend(doc[2])
                doc = doc[0]
            new_lines.append(doc)

            try:
                source_files = source_files = get_source_path(object_name, package)
            except (AttributeError, OSError, TypeError):
                # tokenizers obj do NOT have `__module__` attribute & can NOT be used with inspect.getfile
                source_files = None
        else:
            new_lines.append(lines[idx])
            if lines[idx].startswith("```"):
                is_inside_codeblock = not is_inside_codeblock
            if lines[idx].startswith("#") and not is_inside_codeblock:
                idx_last_heading = len(new_lines) - 1
            idx += 1

    new_content = "\n".join(new_lines)

    return (new_content, anchors, source_files, errors) if return_anchors else new_content


def build_mdx_files(package, doc_folder, output_dir, page_info, version_tag_suffix):
    """
    Build the MDX files for a given package.

    Args:
        package (`types.ModuleType`): The package where to look for objects to document.
        doc_folder (`str` or `os.PathLike`): The folder where the doc source files are.
        output_dir (`str` or `os.PathLike`): The folder where to put the files built.
        page_info (`Dict[str, str]`): Some information about the page.
        version_tag_suffix (`str`, *optional*, defaults to `"src/"`):
            Suffix to add after the version tag (e.g. 1.3.0 or main) in the documentation links.
            For example, the default `"src/"` suffix will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/src/`.
            For example, `version_tag_suffix=""` will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/`.
    """
    doc_folder = Path(doc_folder)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    anchor_mapping = {}
    source_files_mapping = {}

    if "package_name" not in page_info:
        page_info["package_name"] = package.__name__

    all_files = list(doc_folder.glob("**/*"))
    all_errors = []
    for file in tqdm(all_files, desc="Building the MDX files"):
        new_anchors = None
        errors = None
        page_info["path"] = file
        try:
            if file.suffix in [".md", ".mdx"]:
                dest_file = output_dir / (file.with_suffix(".mdx").relative_to(doc_folder))
                page_info["page"] = file.with_suffix(".html").relative_to(doc_folder).as_posix()
                os.makedirs(dest_file.parent, exist_ok=True)
                with open(file, "r", encoding="utf-8-sig") as reader:
                    content = reader.read()
                content = convert_md_to_mdx(content, page_info)
                content = resolve_open_in_colab(content, page_info)
                content, new_anchors, source_files, errors = resolve_autodoc(
                    content, package, return_anchors=True, page_info=page_info, version_tag_suffix=version_tag_suffix
                )
                if source_files is not None:
                    source_files_mapping[source_files] = str(file)
                with open(dest_file, "w", encoding="utf-8") as writer:
                    writer.write(content)
                # Make sure we clean up for next page.
                del page_info["page"]
            elif file.suffix in [".rst"]:
                dest_file = output_dir / (file.with_suffix(".mdx").relative_to(doc_folder))
                page_info["page"] = file.with_suffix(".html").relative_to(doc_folder)
                os.makedirs(dest_file.parent, exist_ok=True)
                with open(file, "r", encoding="utf-8") as reader:
                    content = reader.read()
                content = convert_rst_to_mdx(content, page_info)
                content = resolve_open_in_colab(content, page_info)
                content, new_anchors, source_files, errors = resolve_autodoc(
                    content, package, return_anchors=True, page_info=page_info, version_tag_suffix=version_tag_suffix
                )
                if source_files is not None:
                    source_files_mapping[source_files] = str(file)
                with open(dest_file, "w", encoding="utf-8") as writer:
                    writer.write(content)
                # Make sure we clean up for next page.
                del page_info["page"]
            elif file.is_file() and "__" not in str(file):
                # __ is a reserved svelte file/folder prefix
                dest_file = output_dir / (file.relative_to(doc_folder))
                os.makedirs(dest_file.parent, exist_ok=True)
                shutil.copy(file, dest_file)

        except Exception as e:
            raise type(e)(f"There was an error when converting {file} to the MDX format.\n" + e.args[0]) from e

        if new_anchors is not None:
            page_name = str(file.with_suffix("").relative_to(doc_folder))
            for anchor in new_anchors:
                if isinstance(anchor, tuple):
                    anchor_mapping.update({a: f"{page_name}#{anchor[0]}" for a in anchor[1:]})
                    anchor = anchor[0]
                anchor_mapping[anchor] = page_name

        if errors is not None:
            all_errors.extend(errors)

    if len(all_errors) > 0:
        raise ValueError(
            "The deployment of the documentation will fail because of the following errors:\n" + "\n".join(all_errors)
        )

    return anchor_mapping, source_files_mapping


def resolve_links(doc_folder, package, mapping, page_info):
    """
    Resolve links of the form [`SomeClass`] to the link in the documentation to `SomeClass` for all files in a
    folder.

    Args:
        doc_folder (`str` or `os.PathLike`): The folder in which to look for files.
        package (`types.ModuleType`): The package in which to search objects for.
        mapping (`Dict[str, str]`): The map from anchor names of objects to their page in the documentation.
        page_info (`Dict[str, str]`): Some information about the page.
    """
    doc_folder = Path(doc_folder)
    all_files = list(doc_folder.glob("**/*.mdx"))
    for file in tqdm(all_files, desc="Resolving internal links"):
        with open(file, "r", encoding="utf-8") as reader:
            content = reader.read()
        content = resolve_links_in_text(content, package, mapping, page_info)
        with open(file, "w", encoding="utf-8") as writer:
            writer.write(content)


def build_notebooks(doc_folder, notebook_dir, package=None, mapping=None, page_info=None):
    """
    Build the notebooks associated to the MDX files in the documentation with an [[open-in-colab]] marker.

    Args:
        doc_folder (`str` or `os.PathLike`): The folder where the doc source files are.
        notebook_dir_dir (`str` or `os.PathLike`): Where to save the generated notebooks
        package (`types.ModuleType`, *optional*):
            The package in which to search objects for (needs to be passed to resolve doc links).
        mapping (`Dict[str, str]`, *optional*):
            The map from anchor names of objects to their page in the documentation (needs to be passed to resolve doc
            links).
        page_info (`Dict[str, str]`, *optional*):
            Some information about the page (needs to be passed to resolve doc links).
    """
    doc_folder = Path(doc_folder)

    if "package_name" not in page_info:
        page_info["package_name"] = package.__name__

    mdx_files = list(doc_folder.glob("**/*.mdx"))
    for file in tqdm(mdx_files, desc="Building the notebooks"):
        with open(file, "r", encoding="utf-8") as f:
            if "[[open-in-colab]]" not in f.read():
                continue
        try:
            page_info["page"] = file.with_suffix(".html").relative_to(doc_folder)
            generate_notebooks_from_file(file, notebook_dir, package=package, mapping=mapping, page_info=page_info)
            # Make sure we clean up for next page.
            del page_info["page"]

        except Exception as e:
            raise type(e)(f"There was an error when converting {file} to a notebook.\n" + e.args[0]) from e


def build_doc(
    package_name,
    doc_folder,
    output_dir,
    clean=True,
    version="main",
    version_tag="main",
    language="en",
    notebook_dir=None,
    is_python_module=False,
    watch_mode=False,
    version_tag_suffix="src/",
    repo_owner="huggingface",
    repo_name=None,
):
    """
    Build the documentation of a package.

    Args:
        package_name (`str`): The name of the package.
        doc_folder (`str` or `os.PathLike`): The folder in which the source documentation of the package is.
        output_dir (`str` or `os.PathLike`):
            The folder in which to put the built documentation. Will be created if it does not exist.
        clean (`bool`, *optional*, defaults to `True`):
            Whether or not to delete the content of the `output_dir` if that directory exists.
        version (`str`, *optional*, defaults to `"main"`): The name of the version of the doc.
        version_tag (`str`, *optional*, defaults to `"main"`): The name of the version tag (on GitHub) of the doc.
        language (`str`, *optional*, defaults to `"en"`): The language of the doc.
        notebook_dir (`str` or `os.PathLike`, *optional*):
            If provided, where to save the notebooks generated from the doc file with an [[open-in-colab]] marker.
        is_python_module (`bool`, *optional*, defaults to `False`):
            Whether the docs being built are for python module. (For example, HF Course is not a python module).
        watch_mode (`bool`, *optional*, default to `False`):
            If `True`, disables the toc tree check and sphinx objects.inv builds since they are not needed
            when this mode is active.
        version_tag_suffix (`str`, *optional*, defaults to `"src/"`):
            Suffix to add after the version tag (e.g. 1.3.0 or main) in the documentation links.
            For example, the default `"src/"` suffix will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/src/`.
            For example, `version_tag_suffix=""` will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/`.
        repo_owner (`str`, *optional*, defaults to `"huggingface"`):
            The owner of the repository on GitHub. In most cases, this is `"huggingface"`. However, for the `timm` library, the owner is `"rwightman"`.
        repo_name (`str`, *optional*, defaults to `package_name`):
            The name of the repository on GitHub. In most cases, this is the same as `package_name`. However, for the `timm` library, the name is `"pytorch-image-models"` instead of `"timm"`.
    """
    page_info = {
        "version": version,
        "version_tag": version_tag,
        "language": language,
        "package_name": package_name,
        "repo_owner": repo_owner,
        "repo_name": repo_name if repo_name is not None else package_name,
    }
    if clean and Path(output_dir).exists():
        shutil.rmtree(output_dir)

    read_doc_config(doc_folder)

    package = importlib.import_module(package_name) if is_python_module else None
    anchors_mapping, source_files_mapping = build_mdx_files(
        package, doc_folder, output_dir, page_info, version_tag_suffix=version_tag_suffix
    )
    if not watch_mode:
        sphinx_refs = check_toc_integrity(doc_folder, output_dir)
        sphinx_refs.extend(convert_anchors_mapping_to_sphinx_format(anchors_mapping, package))
    if is_python_module:
        if not watch_mode:
            build_sphinx_objects_ref(sphinx_refs, output_dir, page_info)
        resolve_links(output_dir, package, anchors_mapping, page_info)

    if notebook_dir is not None:
        if clean and Path(notebook_dir).exists():
            for nb_file in Path(notebook_dir).glob("**/*.ipynb"):
                os.remove(nb_file)
        build_notebooks(doc_folder, notebook_dir, package=package, mapping=anchors_mapping, page_info=page_info)

    return source_files_mapping


def check_toc_integrity(doc_folder, output_dir):
    """
    Checks all the MDX files obtained after building the documentation are present in the table of contents.

    Args:
        doc_folder (`str` or `os.PathLike`): The folder where the source files of the documentation lie.
        output_dir (`str` or `os.PathLike`): The folder where the doc is built.
    """
    output_dir = Path(output_dir)
    doc_files = [str(f.relative_to(output_dir).with_suffix("")) for f in output_dir.glob("**/*.mdx")]

    toc_file = Path(doc_folder) / "_toctree.yml"
    with open(toc_file, "r", encoding="utf-8") as f:
        toc = yaml.safe_load(f.read())

    toc_sections = []
    sphinx_refs = []
    # We don't just loop directly in toc as we will add more into it as we un-nest things.
    while len(toc) > 0:
        part = toc.pop(0)
        if "local" in part:
            toc_sections.append(part["local"])
        if "sections" not in part:
            continue
        toc_sections.extend([sec["local"] for sec in part["sections"] if "local" in sec])
        for sec in part["sections"]:
            if "local_fw" in sec:
                toc_sections.extend(sec["local_fw"].values())
        # There should be one sphinx ref per page
        for sec in part["sections"]:
            if "local" in sec:
                sphinx_refs.append(f"{sec['local']} std:doc -1 {sec['local']} {sec['title']}")
        # Toc has some nested sections in the API doc for instance, so we recurse.
        toc.extend([sec for sec in part["sections"] if "sections" in sec])

    # normalize paths to current OS
    toc_sections = [str(Path(path)) for path in toc_sections]
    files_not_in_toc = [f for f in doc_files if f not in toc_sections]
    doc_config = get_doc_config()
    disable_toc_check = getattr(doc_config, "disable_toc_check", False)
    if len(files_not_in_toc) > 0 and not disable_toc_check:
        message = "\n".join([f"- {f}" for f in files_not_in_toc])
        raise RuntimeError(
            "The following files are not present in the table of contents:\n" + message + f"\nAdd them to {toc_file}."
        )

    files_not_exist = [f for f in toc_sections if f not in doc_files]
    if len(files_not_exist) > 0:
        message = "\n".join([f"- {f}" for f in files_not_exist])
        raise RuntimeError(
            "The following files are present in the table of contents but do not exist:\n"
            + message
            + f"\nRemove them from {toc_file}."
        )

    return sphinx_refs


def convert_anchors_mapping_to_sphinx_format(anchors_mapping, package):
    """
    Convert the anchor mapping to the format expected by sphinx for the `objects.inv` file.

    Args:
        anchors_mapping (Dict[`str`, `str`]):
            The mapping between anchors for objects in the doc and their location in the doc.
        package (`types.ModuleType`):
            The package in which to search objects for.
    """
    sphinx_refs = []
    for anchor, url in anchors_mapping.items():
        obj = find_object_in_package(anchor, package)
        if isinstance(obj, property):
            obj = obj.fget

        # Object type
        if isinstance(obj, type):
            obj_type = "py:class"
        elif hasattr(obj, "__name__") and hasattr(obj, "__qualname__"):
            obj_type = "py:method" if obj.__name__ != obj.__qualname__ else "py:function"
        else:
            # Default to function (this part is never hit when building the docs for Transformers and Datasets)
            # so it's just to be extra defensive
            obj_type = "py:function"

        if "#" in url:
            sphinx_refs.append(f"{anchor} {obj_type} 1 {url} -")
        else:
            sphinx_refs.append(f"{anchor} {obj_type} 1 {url}#$ -")

    return sphinx_refs


def build_sphinx_objects_ref(sphinx_refs, output_dir, page_info):
    """
    Saves the sphinx references in an `objects.inv` file that can then be used by other documentations powered by
    sphinx to link to objects in the generated doc.

    Args:
        sphinx_refs (`List[str]`): The list of all references, in the format expected by sphinx.
        output_dir (`str` or `os.PathLike`): The folder where the doc is built.
        page_info (`Dict[str, str]`): Some information about the doc.
    """
    intro = [
        "# Sphinx inventory version 2\n",
        f"# Project: {page_info['package_name']}\n",
        f"# Version: {page_info['version']}\n",
        "# The remainder of this file is compressed using zlib.\n",
    ]
    lines = [str.encode(line) for line in intro]

    data = "\n".join(sorted(sphinx_refs)) + "\n"
    data = zlib.compress(str.encode(data))

    with open(Path(output_dir) / "objects.inv", "wb") as f:
        f.writelines(lines)
        f.write(data)
