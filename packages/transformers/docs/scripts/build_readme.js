import fs from "node:fs";
import path from "node:path";
import url from "node:url";

import { DOCS_BASE_URL, buildApiSymbolLinks } from "./lib/api-links.mjs";
import { loadProject } from "./lib/load.mjs";

const scriptFile = url.fileURLToPath(import.meta.url);
const docsDir = path.dirname(path.dirname(scriptFile));
const packageRoot = path.dirname(docsDir);

const FILES_TO_INCLUDE = {
  intro: "./docs/snippets/0_introduction.snippet",
  quickTour: "./docs/snippets/1_quick-tour.snippet",
  installation: "./docs/snippets/2_installation.snippet",
  customUsage: "./docs/snippets/3_custom-usage.snippet",
  tasks: "./docs/snippets/4_supported-tasks.snippet",
  models: "./docs/snippets/5_supported-models.snippet",
};

const PIPELINE_API_LINK_PREFIX = `${DOCS_BASE_URL}/api/pipelines#module_pipelines.`;

// Links that should point somewhere other than the direct docs URL. Most are
// README-local anchors or guide/API pages referenced by snippets.
const CUSTOM_LINK_MAP = {
  "/custom_usage#convert-your-models-to-onnx": "#convert-your-models-to-onnx",
  "./api/env": `${DOCS_BASE_URL}/api/env`,
  "./guides/webgpu": `${DOCS_BASE_URL}/guides/webgpu`,
  "./guides/dtypes": `${DOCS_BASE_URL}/guides/dtypes`,
};

function main() {
  const { out } = parseArgs(process.argv.slice(2));
  const { ir, publicNames } = loadProject(packageRoot);
  const apiLinks = buildApiSymbolLinks(ir, publicNames);
  const snippets = Object.fromEntries(
    Object.entries(FILES_TO_INCLUDE).map(([key, file]) => [key, fs.readFileSync(path.join(packageRoot, file), "utf8")]),
  );

  const readme = fixLinks(renderTemplate(snippets), apiLinks);
  fs.writeFileSync(path.resolve(packageRoot, out), readme, "utf8");
}

function parseArgs(args) {
  const outIndex = args.indexOf("--out");
  if (outIndex !== -1 && !args[outIndex + 1]) {
    throw new Error("Expected a path after --out.");
  }

  return {
    out: outIndex === -1 ? "README.md" : args[outIndex + 1],
  };
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

function fixLinks(markdown, apiLinks) {
  // This is not a complete Markdown parser, just the narrow link rewrite
  // needed by the README snippets.
  return markdown.replace(/(?<=\])\((.+?)\)/gm, (_, rawLink) => {
    let link = rawLink;
    if (link in CUSTOM_LINK_MAP) {
      link = CUSTOM_LINK_MAP[link];
    } else if (link.startsWith(PIPELINE_API_LINK_PREFIX)) {
      link = apiLinks.get(link.slice(PIPELINE_API_LINK_PREFIX.length)) ?? link;
    } else if (link.startsWith("/")) {
      link = `${DOCS_BASE_URL}${link}`;
    }
    return `(${link})`;
  });
}

main();
