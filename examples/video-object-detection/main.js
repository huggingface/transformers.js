import './style.css';

import { env, AutoModel, AutoProcessor, RawImage } from '@xenova/transformers';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// Proxy the WASM backend to prevent the UI from freezing
env.backends.onnx.wasm.proxy = true;

// Reference the elements that we will need
const status = document.getElementById('status');
const container = document.getElementById('container');
const overlay = document.getElementById('overlay');
const canvas = document.getElementById('canvas');
const video = document.getElementById('video');
const thresholdSlider = document.getElementById('threshold');
const thresholdLabel = document.getElementById('threshold-value');
const sizeSlider = document.getElementById('size');
const sizeLabel = document.getElementById('size-value');
const scaleSlider = document.getElementById('scale');
const scaleLabel = document.getElementById('scale-value');

function setStreamSize(width, height) {
    video.width = canvas.width = Math.round(width);
    video.height = canvas.height = Math.round(height);
}

status.textContent = 'Loading model...';

// Load model and processor
const model_id = 'Xenova/gelan-c_all';
const model = await AutoModel.from_pretrained(model_id);
const processor = await AutoProcessor.from_pretrained(model_id);

// Set up controls
let scale = 0.5;
scaleSlider.addEventListener('input', () => {
    scale = Number(scaleSlider.value);
    setStreamSize(video.videoWidth * scale, video.videoHeight * scale);
    scaleLabel.textContent = scale;
});
scaleSlider.disabled = false;

let threshold = 0.25;
thresholdSlider.addEventListener('input', () => {
    threshold = Number(thresholdSlider.value);
    thresholdLabel.textContent = threshold.toFixed(2);
});
thresholdSlider.disabled = false;

let size = 128;
processor.feature_extractor.size = { shortest_edge: size };
sizeSlider.addEventListener('input', () => {
    size = Number(sizeSlider.value);
    processor.feature_extractor.size = { shortest_edge: size };
    sizeLabel.textContent = size;
});
sizeSlider.disabled = false;

status.textContent = 'Ready';

const COLOURS = [
    "#EF4444", "#4299E1", "#059669",
    "#FBBF24", "#4B52B1", "#7B3AC2",
    "#ED507A", "#1DD1A1", "#F3873A",
    "#4B5563", "#DC2626", "#1852B4",
    "#18A35D", "#F59E0B", "#4059BE",
    "#6027A5", "#D63D60", "#00AC9B",
    "#E64A19", "#272A34"
]

// Render a bounding box and label on the image
function renderBox([xmin, ymin, xmax, ymax, score, id], [w, h]) {
    if (score < threshold) return; // Skip boxes with low confidence

    // Generate a random color for the box
    const color = COLOURS[id % COLOURS.length];

    // Draw the box
    const boxElement = document.createElement('div');
    boxElement.className = 'bounding-box';
    Object.assign(boxElement.style, {
        borderColor: color,
        left: 100 * xmin / w + '%',
        top: 100 * ymin / h + '%',
        width: 100 * (xmax - xmin) / w + '%',
        height: 100 * (ymax - ymin) / h + '%',
    })

    // Draw label
    const labelElement = document.createElement('span');
    labelElement.textContent = `${model.config.id2label[id]} (${(100 * score).toFixed(2)}%)`;
    labelElement.className = 'bounding-box-label';
    labelElement.style.backgroundColor = color;

    boxElement.appendChild(labelElement);
    overlay.appendChild(boxElement);
}

let isProcessing = false;
let previousTime;
const context = canvas.getContext('2d', { willReadFrequently: true });
function updateCanvas() {
    const { width, height } = canvas;
    context.drawImage(video, 0, 0, width, height);

    if (!isProcessing) {
        isProcessing = true;
        (async function () {
            // Read the current frame from the video
            const pixelData = context.getImageData(0, 0, width, height).data;
            const image = new RawImage(pixelData, width, height, 4);

            // Process the image and run the model
            const inputs = await processor(image);
            const { outputs } = await model(inputs);

            // Update UI
            overlay.innerHTML = '';

            const sizes = inputs.reshaped_input_sizes[0].reverse();
            outputs.tolist().forEach(x => renderBox(x, sizes));

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
    window.requestAnimationFrame(updateCanvas);
}).catch((error) => {
    alert(error);
});
