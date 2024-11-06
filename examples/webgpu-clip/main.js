
import {
    AutoTokenizer,
    CLIPTextModelWithProjection,
    AutoProcessor,
    CLIPVisionModelWithProjection,
    RawImage,
    dot,
    softmax,
} from '@xenova/transformers';

import './style.css';

// Reference the elements that we will need
const status = document.getElementById('status');
const container = document.getElementById('container');
const video = document.getElementById('video');
const labelsInput = document.getElementById('labels');
const templateInput = document.getElementById('template');
const overlay = document.getElementById('overlay');

status.textContent = 'Loading model (300MB)...';

// Use fp16 if available, otherwise use fp32
async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter.features.has('shader-f16');
    } catch (e) {
        return false;
    }
}
const dtype = (await hasFp16()) ? 'fp16' : 'fp32';

// Load object detection pipeline
const model_id = 'Xenova/clip-vit-base-patch16';
let tokenizer, text_model, processor, vision_model;
try {
    // Load tokenizer and text model
    tokenizer = await AutoTokenizer.from_pretrained(model_id);
    text_model = await CLIPTextModelWithProjection.from_pretrained(model_id, {
        device: 'webgpu',
        dtype,
    });

    // Load processor and vision model
    processor = await AutoProcessor.from_pretrained(model_id);
    vision_model = await CLIPVisionModelWithProjection.from_pretrained(model_id, {
        device: 'webgpu',
        dtype,
    });

} catch (err) {
    status.textContent = err.message;
    alert(err.message)
    throw err;
}

labelsInput.disabled = false;
templateInput.disabled = false;

status.textContent = 'Ready';

// See `model.logit_scale` parameter of original model
const exp_logit_scale = Math.exp(4.6052);

const IMAGE_SIZE = 224;
const canvas = document.createElement('canvas');
canvas.width = canvas.height = IMAGE_SIZE;
const context = canvas.getContext('2d', { willReadFrequently: true });

let isProcessing = false;
let previousTime;
let textEmbeddings;
let prevTextInputs;
let prevTemplate;
let labels;

function onFrameUpdate() {
    if (!isProcessing) {
        isProcessing = true;
        (async function () {

            // If text inputs have changed, update the embeddings
            if (prevTextInputs !== labelsInput.value || prevTemplate !== templateInput.value) {
                textEmbeddings = null;
                prevTextInputs = labelsInput.value;
                prevTemplate = templateInput.value;
                labels = prevTextInputs.split(/\s*,\s*/).filter(x => x);

                if (labels.length > 0) {
                    const texts = labels.map(x => templateInput.value.replaceAll('{}', x));

                    const text_inputs = tokenizer(texts, { padding: true, truncation: true });

                    // Compute embeddings
                    const { text_embeds } = await text_model(text_inputs);
                    textEmbeddings = text_embeds.normalize().tolist();
                } else {
                    overlay.innerHTML = '';
                }
            }

            if (textEmbeddings) {
                // Read the current frame from the video
                context.drawImage(video, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
                const pixelData = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
                const image = new RawImage(pixelData, IMAGE_SIZE, IMAGE_SIZE, 4);

                const image_inputs = await processor(image);

                // Compute embeddings
                const { image_embeds } = await vision_model(image_inputs);
                const imageEmbedding = image_embeds.normalize().tolist()[0];

                // Compute similarity
                const similarities = textEmbeddings.map(
                    x => dot(x, imageEmbedding) * exp_logit_scale
                );

                const sortedIndices = softmax(similarities)
                    .map((x, i) => [x, i])
                    .sort((a, b) => b[0] - a[0]);

                // Update UI
                overlay.innerHTML = '';
                for (const [score, index] of sortedIndices) {
                    overlay.appendChild(document.createTextNode(`${labels[index]}: ${score.toFixed(2)}`));
                    overlay.appendChild(document.createElement('br'));
                }
            }

            if (previousTime !== undefined) {
                const fps = 1000 / (performance.now() - previousTime);
                status.textContent = `FPS: ${fps.toFixed(2)}`;
            }
            previousTime = performance.now();
            isProcessing = false;
        })();
    }

    window.requestAnimationFrame(onFrameUpdate);
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

    video.width = width;
    video.height = height;

    // Set container width and height depending on the image aspect ratio
    const ar = width / height;
    const [cw, ch] = (ar > 720 / 405) ? [720, 720 / ar] : [405 * ar, 405];
    container.style.width = `${cw}px`;
    container.style.height = `${ch}px`;

    // Start the animation loop
    window.requestAnimationFrame(onFrameUpdate);
}).catch((error) => {
    alert(error);
});
