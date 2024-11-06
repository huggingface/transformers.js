import './style.css';

import { AutoModel, AutoProcessor, RawImage } from '@xenova/transformers';

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter()
        return adapter.features.has('shader-f16')
    } catch (e) {
        return false
    }
}

// Reference the elements that we will need
const status = document.getElementById('status');
const canvas = document.createElement('canvas');
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
const model_id = 'onnx-community/depth-anything-v2-small';

let model;
try {
    model = await AutoModel.from_pretrained(model_id, {
        device: 'webgpu',
        // Use fp16 if available, otherwise use fp32
        dtype: (await hasFp16()) ? 'fp16' : 'fp32',
    });
} catch (err) {
    status.textContent = err.message;
    alert(err.message)
    throw err;
}

const processor = await AutoProcessor.from_pretrained(model_id);

// Set up controls
let size = 504;
processor.feature_extractor.size = { width: size, height: size };
sizeSlider.addEventListener('input', () => {
    size = Number(sizeSlider.value);
    processor.feature_extractor.size = { width: size, height: size };
    sizeLabel.textContent = size;
});
sizeSlider.disabled = false;

let scale = 0.4;
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

            // Predict depth map
            const { predicted_depth } = await model(inputs);
            const data = predicted_depth.data;
            const [bs, oh, ow] = predicted_depth.dims;

            // Normalize the depth map
            let min = Infinity;
            let max = -Infinity;
            outputCanvas.width = ow;
            outputCanvas.height = oh;
            for (let i = 0; i < data.length; ++i) {
                const v = data[i];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            const range = max - min;

            const imageData = new Uint8ClampedArray(4 * data.length);
            for (let i = 0; i < data.length; ++i) {
                const offset = 4 * i;
                imageData[offset] = 255; // Set base color to red

                // Set alpha to normalized depth value
                imageData[offset + 3] = 255 * (1 - (data[i] - min) / range);
            }
            const outPixelData = new ImageData(imageData, ow, oh);
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
    { video: { width: 720, height: 720 } }, // Ask for square video
).then((stream) => {
    // Set up the video and canvas elements.
    video.srcObject = stream;
    video.play();

    const videoTrack = stream.getVideoTracks()[0];
    const { width, height } = videoTrack.getSettings();

    setStreamSize(width * scale, height * scale);

    // Start the animation loop
    setTimeout(updateCanvas, 50);
}).catch((error) => {
    alert(error);
});
