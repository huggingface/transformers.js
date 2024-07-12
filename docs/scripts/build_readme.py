
import re
README_TEMPLATE = """

<p align="center">
    <br/>
    <picture> 
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/xenova/transformers.js/assets/26504141/bd047e0f-aca9-4ff7-ba07-c7ca55442bc4" width="500" style="max-width: 100%;">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/xenova/transformers.js/assets/26504141/84a5dc78-f4ea-43f4-96f2-b8c791f30a8e" width="500" style="max-width: 100%;">
        <img alt="transformers.js javascript library logo" src="https://github.com/xenova/transformers.js/assets/26504141/84a5dc78-f4ea-43f4-96f2-b8c791f30a8e" width="500" style="max-width: 100%;">
    </picture>
    <br/>
</p>

<p align="center">
    <a href="https://www.npmjs.com/package/@xenova/transformers"><img alt="NPM" src="https://img.shields.io/npm/v/@xenova/transformers"></a>
    <a href="https://www.npmjs.com/package/@xenova/transformers"><img alt="NPM Downloads" src="https://img.shields.io/npm/dw/@xenova/transformers"></a>
    <a href="https://www.jsdelivr.com/package/npm/@xenova/transformers"><img alt="jsDelivr Hits" src="https://img.shields.io/jsdelivr/npm/hw/@xenova/transformers"></a>
    <a href="https://github.com/xenova/transformers.js/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/xenova/transformers.js?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers.js/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers.js/index.svg?down_color=red&down_message=offline&up_message=online"></a>
</p>

{intro}

## Quick tour

{quick_tour}

## Installation

{installation}

## Examples

{examples}

## Custom usage

{custom_usage}

## Supported tasks/models

Here is the list of all tasks and architectures currently supported by Transformers.js.
If you don't see your task/model listed here or it is not yet supported, feel free
to open up a feature request [here](https://github.com/xenova/transformers.js/issues/new/choose).

To find compatible models on the Hub, select the "transformers.js" library tag in the filter menu (or visit [this link](https://huggingface.co/models?library=transformers.js)).
You can refine your search by selecting the task you're interested in (e.g., [text-classification](https://huggingface.co/models?pipeline_tag=text-classification&library=transformers.js)).

{tasks}

{models}
"""


FILES_TO_INCLUDE = dict(
    intro='./docs/snippets/0_introduction.snippet',
    quick_tour='./docs/snippets/1_quick-tour.snippet',
    installation='./docs/snippets/2_installation.snippet',
    examples='./docs/snippets/3_examples.snippet',
    custom_usage='./docs/snippets/4_custom-usage.snippet',
    tasks='./docs/snippets/5_supported-tasks.snippet',
    models='./docs/snippets/6_supported-models.snippet',
)

DOCS_BASE_URL = 'https://huggingface.co/docs/transformers.js'

# Map of custom links to replace, typically used for links to other sections of the README.
CUSTOM_LINK_MAP = {
    '/custom_usage#convert-your-models-to-onnx': '#convert-your-models-to-onnx',
    './api/env': DOCS_BASE_URL + '/api/env',
}


def main():

    file_data = {}
    for key, file_path in FILES_TO_INCLUDE.items():
        with open(file_path, encoding='utf-8') as f:
            file_data[key] = f.read()

    # Fix links:
    # NOTE: This regex does not match all markdown links, but works for the ones we need to replace.
    LINK_RE = r'(?<=\])\((.+?)\)'

    def replace_fn(match):
        link = match.group(1)

        if link in CUSTOM_LINK_MAP:
            link = CUSTOM_LINK_MAP[link]

        elif link.startswith('/'):
            # Link to docs
            link = DOCS_BASE_URL + link

        elif link.startswith('./'):
            # Relative link to file
            pass

        elif link.startswith('http'):
            # Link to external site
            pass

        return f'({link})'

    result = README_TEMPLATE.format(**file_data)
    result = re.sub(LINK_RE, replace_fn, result, 0, re.MULTILINE)

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    main()
