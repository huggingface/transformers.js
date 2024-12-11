// offscreen.js - Handles requests from the background service worker, runs the model, then sends back a response

// Registering this listener when the script is first executed ensures that the
// offscreen document will be able to receive messages when the promise returned
// by `offscreen.createDocument()` resolves.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("sender", sender)
    console.log('message', message)
    if (message.target !== 'offscreen') {
        return false;
    }
    switch (message.type) {
        case 'classify':
            sendToSandboxForInference(message.data).then(result => {
                sendResponse({ type: 'result', result });
            });
            return true;
        default:
            console.warn(`Unexpected message type received: '${message.type}'.`);
            return false;
    }
});

// The sandbox is neccesary because the ONNX Runtime loads in a way that violates the typical Chrome extension Content Security Policy
// Using a MessageChannel allows us to use an async request/response pattern: https://advancedweb.hu/how-to-use-async-await-with-postmessage/
const sendToSandboxForInference = input => new Promise((res, rej) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = ({ data }) => {
        channel.port1.close();
        if (data.error) {
            rej(data.error);
        } else {
            res(data.result);
        }
    };
    document.getElementById('sandbox').contentWindow.postMessage(input, '*', [channel.port2]);
});