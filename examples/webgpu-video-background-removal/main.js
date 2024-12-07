import './style.css';

import { env, AutoModel, AutoProcessor, RawImage } from '@xenova/transformers';

env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
env.backends.onnx.wasm.numThreads = 1;

// Reference the elements that we will need
const status = document.getElementById('status');
const container = document.getElementById('container');
const canvas = document.getElementById('canvas');
const outputCanvas = document.getElementById('output-canvas');
const video = document.getElementById('video');
const sizeSlider = document.getElementById('size');
const sizeLabel = document.getElementById('size-value');
const scaleSlider = document.getElementById('scale');
const scaleLabel = document.getElementById('scale-value');

function setStreamSize(width, height) {
    video.width = outputCanvas.width = canvas.width = Math.round(width);
    video.height = outputCanvas.height = canvas.height = Math.round(height);
}

status.textContent = 'Loading model...';

// Load model and processor
const model_id = 'Xenova/modnet';
let model;
try {
    model = await AutoModel.from_pretrained(model_id, {
        device: 'webgpu',
        dtype: 'fp32', // TODO: add fp16 support
    });
} catch (err) {
    status.textContent = err.message;
    alert(err.message)
    throw err;
}

const processor = await AutoProcessor.from_pretrained(model_id);

// Set up controls
let size = 256;
processor.feature_extractor.size = { shortest_edge: size };
sizeSlider.addEventListener('input', () => {
    size = Number(sizeSlider.value);
    processor.feature_extractor.size = { shortest_edge: size };
    sizeLabel.textContent = size;
});
sizeSlider.disabled = false;

let scale = 0.5;
scaleSlider.addEventListener('input', () => {
    scale = Number(scaleSlider.value);
    setStreamSize(video.videoWidth * scale, video.videoHeight * scale);
    scaleLabel.textContent = scale;
});
scaleSlider.disabled = false;

status.textContent = 'Ready';

let isProcessing = false;
let previousTime;
const context = canvas.getContext('2d', { willReadFrequently: true });
const outputContext = outputCanvas.getContext('2d', { willReadFrequently: true });
function updateCanvas() {
    const { width, height } = canvas;

    if (!isProcessing) {
        isProcessing = true;
        (async function () {
            // Read the current frame from the video
            context.drawImage(video, 0, 0, width, height);
            const currentFrame = context.getImageData(0, 0, width, height);
            const image = new RawImage(currentFrame.data, width, height, 4);

            // Pre-process image
            const inputs = await processor(image);

            // Predict alpha matte
            const { output } = await model({ input: inputs.pixel_values });

            const mask = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(width, height);

            // Update alpha channel
            const outPixelData = currentFrame;
            for (let i = 0; i < mask.data.length; ++i) {
                outPixelData.data[4 * i + 3] = mask.data[i];
            }
            outputContext.putImageData(outPixelData, 0, 0);

            if (previousTime !== undefined) {
                const fps = 1000 / (performance.now() - previousTime);
                status.textContent = `FPS: ${fps.toFixed(2)}`;
            }
            previousTime = performance.now();

            isProcessing = false;
        })();
    }

    window.requestAnimationFrame(updateCanvas);
}

// Start the video stream
navigator.mediaDevices.getUserMedia(
    { video: true }, // Ask for video
).then((stream) => {
    // Set up the video and canvas elements.
    video.srcObject = stream;
    video.play();

    const videoTrack = stream.getVideoTracks()[0];
    const { width, height } = videoTrack.getSettings();

    setStreamSize(width * scale, height * scale);

    // Set container width and height depending on the image aspect ratio
    const ar = width / height;
    const [cw, ch] = (ar > 720 / 405) ? [720, 720 / ar] : [405 * ar, 405];
    container.style.width = `${cw}px`;
    container.style.height = `${ch}px`;

    // Start the animation loop
    setTimeout(updateCanvas, 50);
}).catch((error) => {
    alert(error);
});
