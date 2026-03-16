import { PipelineSingleton } from "./inference.mjs";

// Create generic inference function, which will be reused for the different types of events.
const infer = async (text, message_port) => {
    // Get the pipeline instance. This will load and build the model when run for the first time.
    let model = await PipelineSingleton.getInstance((data) => {
        // Post messages back on the same channel to indicate model loading progress.
        data.type = 'load'
        message_port.postMessage(data)
    });

    // Actually run the model on the input text
    let result = await model(text);
    return result;
};

// input, output, and errors as passed to and from the sandbox via a MessageChannel
window.addEventListener('message', function (event) {
    try {
        infer(event.data, event.ports[0]).then(result => {
            event.ports[0].postMessage({ type: 'result', result });
        })
    } catch (e) {
        event.ports[0].postMessage({ type: 'error', error: e });
    }
}, false)