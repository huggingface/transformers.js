// sidepanel.js - handles the sidepanel UI and starts Worker(s) to run inference

const inputElement = document.getElementById('text');
const outputElement = document.getElementById('output');
const loadingInfoElement = document.getElementById('model_loading_info')

let sandboxLoaded = new Promise((resolve, reject) => {
    document.getElementById('sandbox').addEventListener('load', () => resolve());
})

function show_load_progress(data) {
    const progress_id = `${data.file}_progress`
    switch (data.status) {
        case 'initiate':
            let div = document.createElement('div')
            div.id = progress_id + '_div'
            let label = document.createElement('label')
            label.setAttribute('for', progress_id)
            label.innerText = data.file
            let progress = document.createElement('progress')
            progress.id = progress_id
            div.appendChild(label)
            div.appendChild(progress)
            loadingInfoElement.appendChild(div)
            break
        case 'progress':
            document.getElementById(progress_id).value = data.progress
            break
        case 'done':
            document.getElementById(progress_id + '_div').style.display = 'none'
            break
        case 'ready':
            loadingInfoElement.style.display = 'none'
            break
    }
}

// We load the model and conduct inference in the sandbox.
// The sandbox is neccesary because the ONNX Runtime loads in a way that violates the typical Chrome extension Content Security Policy
// Using a MessageChannel allows us to use an async request/response pattern: https://advancedweb.hu/how-to-use-async-await-with-postmessage/
const sendToSandboxForInference = input => new Promise((res, rej) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = ({ data }) => {
        switch (data.type) {
            case 'load':
                show_load_progress(data)
                break
            case 'result':
                channel.port1.close();
                res(data.result);
                break
            case 'error':
                channel.port1.close();
                rej(data.error);
                break
        }
    };
    document.getElementById('sandbox').contentWindow.postMessage(input, '*', [channel.port2]);
});

const main = async inference_input => {
    inputElement.value = inference_input;
    await sandboxLoaded  // need to wait for ifr
    const result = await sendToSandboxForInference(inference_input)
    outputElement.innerText = JSON.stringify(result, null, 2);
}


// On load, check if there's any pending input saved by the background worker because the sidepanel wasn't open yet
const pending = await chrome.storage.session.get('inference_input')
if (pending.inference_input?.length > 0) {
    main(pending.inference_input)
    chrome.storage.session.remove('inference_input')
}

// Open port and listen for additional input from the background worker once open
const port = chrome.runtime.connect()
port.onMessage.addListener((message, sender, sendResponse) => {
    main(message.text)
})

// Accept input added directly to the text box in the sidepanel UI
inputElement.addEventListener('input', async (event) => {
    main(event.target.value)
});
