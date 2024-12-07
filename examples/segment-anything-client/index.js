
// Reference the elements we will use
const statusLabel = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const example = document.getElementById('example');
const maskCanvas = document.getElementById('mask-output');
const uploadButton = document.getElementById('upload-button');
const resetButton = document.getElementById('reset-image');
const clearButton = document.getElementById('clear-points');
const cutButton = document.getElementById('cut-mask');

// State variables
let lastPoints = null;
let isEncoded = false;
let isDecoding = false;
let isMultiMaskMode = false;
let modelReady = false;
let imageDataURI = null;

// Constants
const BASE_URL = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/';
const EXAMPLE_URL = BASE_URL + 'corgi.jpg';

// Create a web worker so that the main (UI) thread is not blocked during inference.
const worker = new Worker(
    new URL('./worker.js', import.meta.url),
    { type: 'module' }
);

// Preload star and cross images to avoid lag on first click
const star = new Image();
star.src = BASE_URL + 'star-icon.png';
star.className = 'icon';

const cross = new Image();
cross.src = BASE_URL + 'cross-icon.png';
cross.className = 'icon';

// Set up message handler
worker.addEventListener('message', (e) => {
    const { type, data } = e.data;
    if (type === 'ready') {
        modelReady = true;
        statusLabel.textContent = 'Ready';

    } else if (type === 'decode_result') {
        isDecoding = false;

        if (!isEncoded) {
            return; // We are not ready to decode yet
        }

        if (!isMultiMaskMode && lastPoints) {
            // Perform decoding with the last point
            decode();
            lastPoints = null;
        }

        const { mask, scores } = data;

        // Update canvas dimensions (if different)
        if (maskCanvas.width !== mask.width || maskCanvas.height !== mask.height) {
            maskCanvas.width = mask.width;
            maskCanvas.height = mask.height;
        }

        // Create context and allocate buffer for pixel data
        const context = maskCanvas.getContext('2d');
        const imageData = context.createImageData(maskCanvas.width, maskCanvas.height);

        // Select best mask
        const numMasks = scores.length; // 3
        let bestIndex = 0;
        for (let i = 1; i < numMasks; ++i) {
            if (scores[i] > scores[bestIndex]) {
                bestIndex = i;
            }
        }
        statusLabel.textContent = `Segment score: ${scores[bestIndex].toFixed(2)}`;

        // Fill mask with colour
        const pixelData = imageData.data;
        for (let i = 0; i < pixelData.length; ++i) {
            if (mask.data[numMasks * i + bestIndex] === 1) {
                const offset = 4 * i;
                pixelData[offset] = 0;       // red
                pixelData[offset + 1] = 114; // green
                pixelData[offset + 2] = 189; // blue
                pixelData[offset + 3] = 255; // alpha
            }
        }

        // Draw image data to context
        context.putImageData(imageData, 0, 0);

    } else if (type === 'segment_result') {
        if (data === 'start') {
            statusLabel.textContent = 'Extracting image embedding...';
        } else {
            statusLabel.textContent = 'Embedding extracted!';
            isEncoded = true;
        }
    }
});

function decode() {
    isDecoding = true;
    worker.postMessage({ type: 'decode', data: lastPoints });
}

function clearPointsAndMask() {
    // Reset state
    isMultiMaskMode = false;
    lastPoints = null;

    // Remove points from previous mask (if any)
    document.querySelectorAll('.icon').forEach(e => e.remove());

    // Disable cut button
    cutButton.disabled = true;

    // Reset mask canvas
    maskCanvas.getContext('2d').clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}
clearButton.addEventListener('click', clearPointsAndMask);

resetButton.addEventListener('click', () => {
    // Update state
    isEncoded = false;
    imageDataURI = null;

    // Indicate to worker that we have reset the state
    worker.postMessage({ type: 'reset' });

    // Clear points and mask (if present)
    clearPointsAndMask();

    // Update UI
    cutButton.disabled = true;
    imageContainer.style.backgroundImage = 'none';
    uploadButton.style.display = 'flex';
    statusLabel.textContent = 'Ready';
});

function segment(data) {
    // Update state
    isEncoded = false;
    if (!modelReady) {
        statusLabel.textContent = 'Loading model...';
    }
    imageDataURI = data;

    // Update UI
    imageContainer.style.backgroundImage = `url(${data})`;
    uploadButton.style.display = 'none';
    cutButton.disabled = true;

    // Instruct worker to segment the image
    worker.postMessage({ type: 'segment', data });
}

// Handle file selection
fileUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();

    // Set up a callback when the file is loaded
    reader.onload = e2 => segment(e2.target.result);

    reader.readAsDataURL(file);
});

example.addEventListener('click', (e) => {
    e.preventDefault();
    segment(EXAMPLE_URL);
});

function addIcon({ point, label }) {
    const icon = (label === 1 ? star : cross).cloneNode();
    icon.style.left = `${point[0] * 100}%`;
    icon.style.top = `${point[1] * 100}%`;
    imageContainer.appendChild(icon);
}

// Attach hover event to image container
imageContainer.addEventListener('mousedown', e => {
    if (e.button !== 0 && e.button !== 2) {
        return; // Ignore other buttons
    }
    if (!isEncoded) {
        return; // Ignore if not encoded yet
    }
    if (!isMultiMaskMode) {
        lastPoints = [];
        isMultiMaskMode = true;
        cutButton.disabled = false;
    }

    const point = getPoint(e);
    lastPoints.push(point);

    // add icon
    addIcon(point);

    decode();
});


// Clamp a value inside a range [min, max]
function clamp(x, min = 0, max = 1) {
    return Math.max(Math.min(x, max), min)
}

function getPoint(e) {
    // Get bounding box
    const bb = imageContainer.getBoundingClientRect();

    // Get the mouse coordinates relative to the container
    const mouseX = clamp((e.clientX - bb.left) / bb.width);
    const mouseY = clamp((e.clientY - bb.top) / bb.height);

    return {
        point: [mouseX, mouseY],
        label: e.button === 2 // right click
            ? 0  // negative prompt
            : 1, // positive prompt
    }
}

// Do not show context menu on right click
imageContainer.addEventListener('contextmenu', e => {
    e.preventDefault();
});

// Attach hover event to image container
imageContainer.addEventListener('mousemove', e => {
    if (!isEncoded || isMultiMaskMode) {
        // Ignore mousemove events if the image is not encoded yet,
        // or we are in multi-mask mode
        return;
    }
    lastPoints = [getPoint(e)];

    if (!isDecoding) {
        decode(); // Only decode if we are not already decoding
    }
});

// Handle cut button click
cutButton.addEventListener('click', () => {
    const [w, h] = [maskCanvas.width, maskCanvas.height];

    // Get the mask pixel data
    const maskContext = maskCanvas.getContext('2d');
    const maskPixelData = maskContext.getImageData(0, 0, w, h);

    // Load the image
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = async () => {
        // Create a new canvas to hold the image
        const imageCanvas = new OffscreenCanvas(w, h);
        const imageContext = imageCanvas.getContext('2d');
        imageContext.drawImage(image, 0, 0, w, h);
        const imagePixelData = imageContext.getImageData(0, 0, w, h);

        // Create a new canvas to hold the cut-out
        const cutCanvas = new OffscreenCanvas(w, h);
        const cutContext = cutCanvas.getContext('2d');
        const cutPixelData = cutContext.getImageData(0, 0, w, h);

        // Copy the image pixel data to the cut canvas
        for (let i = 3; i < maskPixelData.data.length; i += 4) {
            if (maskPixelData.data[i] > 0) {
                for (let j = 0; j < 4; ++j) {
                    const offset = i - j;
                    cutPixelData.data[offset] = imagePixelData.data[offset];
                }
            }
        }
        cutContext.putImageData(cutPixelData, 0, 0);

        // Download image 
        const link = document.createElement('a');
        link.download = 'image.png';
        link.href = URL.createObjectURL(await cutCanvas.convertToBlob());
        link.click();
        link.remove();
    }
    image.src = imageDataURI;
});
