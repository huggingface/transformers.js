// Based on [this tutorial](https://github.com/jsdoc2md/jsdoc-to-markdown/wiki/How-to-create-one-output-file-per-class).

import fs from 'fs';
import path from 'path';
import url from 'url';

import jsdoc2md from 'jsdoc-to-markdown';

const docs = path.dirname(path.dirname(url.fileURLToPath(import.meta.url)));
const root = path.dirname(docs);

// jsdoc config file
const conf = path.join(docs, 'jsdoc-conf.json');

// input and output paths
const inputFile = path.join(root, '/src/**/*.js');
const outputDir = path.join(root, '/docs/source/api/');


// get template data
const templateData = await jsdoc2md.getTemplateData({
    files: inputFile,
    configure: conf
})

// reduce templateData to an array of module names
const moduleNames = templateData.reduce(
    (moduleNames, identifier) => {
        if (identifier.kind === 'module') {
            moduleNames.push(identifier.name)
        }
        return moduleNames
    }, []
)

// create a documentation file for each module
for (const moduleName of moduleNames) {
    const template = `{{#module name="${moduleName}"}}{{>docs}}{{/module}}`;
    console.log(`rendering ${moduleName}, template: ${template}`);
    let output = await jsdoc2md.render({
        'data': templateData,
        'template': template,
        'heading-depth': 1,
        'no-gfm': true,
        'name-format': 'backticks',
        'no-cache': true,
        'separators': true,
        'configure': conf,
    });

    // Post-processing
    output = output.replace(/(^#+\s.+)/gm, '$1\n'); // Add new line after each header

    // Replace all generated marker names with ids (for linking), and add group class
    output = output.replace(/<a name="(\S+)"><\/a>/g, '<a id="$1" class="group"></a>');

    // Unescape some of the characters which jsdoc2md escapes:
    // TODO: May need to extend this list
    output = output.replace(/\\([|_&*])/gm, '$1');

    output = output.replaceAll('new exports.', 'new ');

    let outputPath = path.resolve(outputDir, `${moduleName}.md`);

    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, output);

}
