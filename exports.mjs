import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

const classesDir = './src/models/classes';

// Get all files in the classes directory
const files = await readdir(classesDir);

// Filter only .js files
const jsFiles = files.filter(file => file.endsWith('.js'));

// For each file, read it and extract exported classes
for (const file of jsFiles.sort()) {
    const filePath = join(classesDir, file);
    const content = await readFile(filePath, 'utf-8');

    // Extract all exported class names
    const classMatches = content.matchAll(/export class (\w+)/g);
    const classes = Array.from(classMatches, m => m[1]);

    if (classes.length > 0) {
        console.log(`import { ${classes.join(', ')} } from './classes/${file}';`);
    }
}
