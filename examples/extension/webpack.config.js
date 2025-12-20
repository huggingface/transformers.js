
import path from 'path';
import { fileURLToPath } from 'url';

import HtmlWebpackPlugin from 'html-webpack-plugin';
import CopyPlugin from 'copy-webpack-plugin';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const config = {
    mode: 'development',
    devtool: 'inline-source-map',
    entry: {
        background: {
            import: './src/background.js',
            chunkLoading: `import-scripts`,
        },
        popup: './src/popup.js',
        content: './src/content.js',
    },
    output: {
        path: path.resolve(__dirname, 'build'),
        filename: '[name].js',
    },
    resolve: {
        fallback: {
            "fs": false,
            "path": false,
            "url": false,
        }
    },
    module: {
        parser: {
            javascript: {
                importMeta: false,
            },
        },
    },
    ignoreWarnings: [
        {
            module: /node_modules\/@huggingface\/transformers/,
            message: /Critical dependency: Accessing import\.meta directly is unsupported/,
        },
    ],
    plugins: [
        new HtmlWebpackPlugin({
            template: './src/popup.html',
            filename: 'popup.html',
        }),
        new CopyPlugin({
            patterns: [
                {
                    from: "public",
                    to: "." // Copies to build folder
                },
                {
                    from: "src/popup.css",
                    to: "popup.css"
                }
            ],
        })
    ],
};

export default config;
