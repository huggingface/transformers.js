import fs from "node:fs";

const DOCS_BASE_URL = "https://huggingface.co/docs/transformers.js";

const FILES_TO_INCLUDE = {
  intro: "./docs/snippets/0_introduction.snippet",
  quickTour: "./docs/snippets/1_quick-tour.snippet",
  installation: "./docs/snippets/2_installation.snippet",
  customUsage: "./docs/snippets/3_custom-usage.snippet",
  tasks: "./docs/snippets/4_supported-tasks.snippet",
  models: "./docs/snippets/5_supported-models.snippet",
};

// Links that should point somewhere other than the direct docs URL.
const CUSTOM_LINK_MAP = {
  "/custom_usage#convert-your-models-to-onnx": "#convert-your-models-to-onnx",
  "./api/env": `${DOCS_BASE_URL}/api/env`,
  "./guides/webgpu": `${DOCS_BASE_URL}/guides/webgpu`,
  "./guides/dtypes": `${DOCS_BASE_URL}/guides/dtypes`,
};

function main() {
  const snippets = Object.fromEntries(
    Object.entries(FILES_TO_INCLUDE).map(([key, file]) => [key, fs.readFileSync(file, "utf8")]),
  );

  const readme = fixLinks(renderTemplate(snippets));
  fs.writeFileSync("README.md", readme, "utf8");
}

function renderTemplate({ intro, installation, quickTour, customUsage, tasks, models }) {
  return `

<p align="center">
    <br/>
    <picture> 
        <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/transformersjs-dark.svg" width="500" style="max-width: 100%;">
        <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/transformersjs-light.svg" width="500" style="max-width: 100%;">
        <img alt="transformers.js javascript library logo" src="https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/transformersjs-light.svg" width="500" style="max-width: 100%;">
    </picture>
    <br/>
</p>

<p align="center">
    <a href="https://www.npmjs.com/package/@huggingface/transformers"><img alt="NPM" src="https://img.shields.io/npm/v/@huggingface/transformers"></a>
    <a href="https://www.npmjs.com/package/@huggingface/transformers"><img alt="NPM Downloads" src="https://img.shields.io/npm/dw/@huggingface/transformers"></a>
    <a href="https://www.jsdelivr.com/package/npm/@huggingface/transformers"><img alt="jsDelivr Hits" src="https://img.shields.io/jsdelivr/npm/hw/@huggingface/transformers"></a>
    <a href="https://github.com/huggingface/transformers.js/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/transformers.js?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers.js/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers.js/index.svg?down_color=red&down_message=offline&up_message=online"></a>
</p>

${intro}

## Installation

${installation}

## Quick tour

${quickTour}

## Custom usage

${customUsage}

## Supported tasks/models

Here is the list of all tasks and architectures currently supported by Transformers.js. If you don't see your task/model listed here or it is not yet supported, feel free to open up a feature request [here](https://github.com/huggingface/transformers.js/issues/new/choose).

To find compatible models on the Hub, select the "transformers.js" library tag in the filter menu (or visit [this link](https://huggingface.co/models?library=transformers.js)). You can refine your search by selecting the task you're interested in (e.g., [text-classification](https://huggingface.co/models?pipeline_tag=text-classification&library=transformers.js)).

${tasks}

${models}
`;
}

function fixLinks(markdown) {
  return markdown.replace(/(?<=\])\((.+?)\)/gm, (_, rawLink) => {
    let link = rawLink;
    if (link in CUSTOM_LINK_MAP) {
      link = CUSTOM_LINK_MAP[link];
    } else if (link.startsWith("/")) {
      link = `${DOCS_BASE_URL}${link}`;
    }
    return `(${link})`;
  });
}

main();
