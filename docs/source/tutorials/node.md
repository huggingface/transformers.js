
# Server-side Inference in Node.js

Although Transformers.js was originally designed to be used in the browser, it's also able to run inference on the server. In this tutorial, we will design a simple Node.js API that uses Transformers.js for sentiment analysis.

We'll also show you how to use the library in both CommonJS and ECMAScript modules, so you can choose the module system that works best for your project:

- [ECMAScript modules (ESM)](#ecmascript-modules-esm) - The official standard format
  to package JavaScript code for reuse. It's the default module system in modern
  browsers, with modules imported using `import` and exported using `export`.
  Fortunately, starting with version 13.2.0, Node.js has stable support of ES modules.
- [CommonJS](#commonjs) - The default module system in Node.js. In this system,
  modules are imported using `require()` and exported using `module.exports`.

<Tip>

Although you can always use the [Python library](https://github.com/huggingface/transformers) for server-side inference, using Transformers.js means that you can write all of your code in JavaScript (instead of having to set up and communicate with a separate Python process).

</Tip>

**Useful links:**
- Source code ([ESM](https://github.com/huggingface/transformers.js/tree/main/examples/node/esm/app.js) or [CommonJS](https://github.com/huggingface/transformers.js/tree/main/examples/node/commonjs/app.js))
- [Documentation](https://huggingface.co/docs/transformers.js) 


## Prerequisites

- [Node.js](https://nodejs.org/en/) version 18+
- [npm](https://www.npmjs.com/) version 9+


## Getting started

Let's start by creating a new Node.js project and installing Transformers.js via [NPM](https://www.npmjs.com/package/@huggingface/transformers):

```bash
npm init -y
npm i @huggingface/transformers
```

Next, create a new file called `app.js`, which will be the entry point for our application. Depending on whether you're using [ECMAScript modules](#ecmascript-modules-esm) or [CommonJS](#commonjs), you will need to do some things differently (see below).

We'll also create a helper class called `MyClassificationPipeline` control the loading of the pipeline. It uses the [singleton pattern](https://en.wikipedia.org/wiki/Singleton_pattern) to lazily create a single instance of the pipeline when `getInstance` is first called, and uses this pipeline for all subsequent calls:


### ECMAScript modules (ESM)

To indicate that your project uses ECMAScript modules, you need to add `"type": "module"` to your `package.json`:

```json
{
  ...
  "type": "module",
  ...
}
```

Next, you will need to add the following imports to the top of `app.js`:

```javascript
import http from 'http';
import querystring from 'querystring';
import url from 'url';
```

Following that, let's import Transformers.js and define the `MyClassificationPipeline` class.

```javascript
import { pipeline, env } from '@huggingface/transformers';

class MyClassificationPipeline {
  static task = 'text-classification';
  static model = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';
  static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      // NOTE: Uncomment this to change the cache directory
      // env.cacheDir = './.cache';

      this.instance = pipeline(this.task, this.model, { progress_callback });
    }

    return this.instance;
  }
}
```

### CommonJS

Start by adding the following imports to the top of `app.js`:

```javascript
const http = require('http');
const querystring = require('querystring');
const url = require('url');
```

Following that, let's import Transformers.js and define the `MyClassificationPipeline` class. Since Transformers.js is an ESM module, we will need to dynamically import the library using the [`import()`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/import) function:

```javascript
class MyClassificationPipeline {
  static task = 'text-classification';
  static model = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';
  static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      // Dynamically import the Transformers.js library
      let { pipeline, env } = await import('@huggingface/transformers');

      // NOTE: Uncomment this to change the cache directory
      // env.cacheDir = './.cache';

      this.instance = pipeline(this.task, this.model, { progress_callback });
    }

    return this.instance;
  }
}
```

## Creating a basic HTTP server

Next, let's create a basic server with the built-in [HTTP](https://nodejs.org/api/http.html#http) module. We will listen for requests made to the server (using the `/classify` endpoint), extract the `text` query parameter, and run this through the pipeline.

```javascript
// Define the HTTP server
const server = http.createServer();
const hostname = '127.0.0.1';
const port = 3000;

// Listen for requests made to the server
server.on('request', async (req, res) => {
  // Parse the request URL
  const parsedUrl = url.parse(req.url);

  // Extract the query parameters
  const { text } = querystring.parse(parsedUrl.query);

  // Set the response headers
  res.setHeader('Content-Type', 'application/json');

  let response;
  if (parsedUrl.pathname === '/classify' && text) {
    const classifier = await MyClassificationPipeline.getInstance();
    response = await classifier(text);
    res.statusCode = 200;
  } else {
    response = { 'error': 'Bad request' }
    res.statusCode = 400;
  }

  // Send the JSON response
  res.end(JSON.stringify(response));
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});

```

<Tip>

Since we use lazy loading, the first request made to the server will also be responsible for loading the pipeline. If you would like to begin loading the pipeline as soon as the server starts running, you can add the following line of code after defining `MyClassificationPipeline`:

```javascript
MyClassificationPipeline.getInstance();
```

</Tip>

To start the server, run the following command:

```bash
node app.js
```

The server should be live at http://127.0.0.1:3000/, which you can visit in your web browser. You should see the following message:

```json
{"error":"Bad request"}
```

This is because we aren't targeting the `/classify` endpoint with a valid `text` query parameter. Let's try again, this time with a valid request. For example, you can visit http://127.0.0.1:3000/classify?text=I%20love%20Transformers.js and you should see:

```json
[{"label":"POSITIVE","score":0.9996721148490906}]
```

Great! We've successfully created a basic HTTP server that uses Transformers.js to classify text.

## (Optional) Customization

### Model caching

By default, the first time you run the application, it will download the model files and cache them on your file system (in `./node_modules/@huggingface/transformers/.cache/`). All subsequent requests will then use this model. You can change the location of the cache by setting `env.cacheDir`. For example, to cache the model in the `.cache` directory in the current working directory, you can add:

```javascript
env.cacheDir = './.cache';
```

### Use local models

If you want to use local model files, you can set `env.localModelPath` as follows:

```javascript
// Specify a custom location for models (defaults to '/models/').
env.localModelPath = '/path/to/models/';
```

You can also disable loading of remote models by setting `env.allowRemoteModels` to `false`:

```javascript
// Disable the loading of remote models from the Hugging Face Hub:
env.allowRemoteModels = false;
```
