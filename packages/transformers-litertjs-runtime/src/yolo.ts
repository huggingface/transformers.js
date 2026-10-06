import { Tensor, type RawImage } from '@huggingface/transformers';
import type { CompiledModel } from '@litertjs/core';
import type { LiteRtModelDescriptor } from './descriptors.js';
import { runVisionModel } from './vision.js';

type Detection = { box: [number, number, number, number]; score: number; classId: number };
type YoloImageTransform = {
    scale: number;
    padX: number;
    padY: number;
    originalWidth: number;
    originalHeight: number;
};
type YoloInputs = { pixel_values: Tensor; image_transform?: YoloImageTransform };
type YoloOutput = { predictions: Tensor; image_transform?: YoloImageTransform };

export async function preprocessYolo(images: RawImage[]): Promise<YoloInputs> {
    if (images.length !== 1) throw new Error('LiteRT.js YOLO currently supports batch size 1.');
    const image = images[0].clone().rgb();
    const scale = Math.min(640 / image.width, 640 / image.height);
    const resizedWidth = Math.round(image.width * scale);
    const resizedHeight = Math.round(image.height * scale);
    const resized = await image.resize(resizedWidth, resizedHeight, { resample: 2 });
    const padX = Math.floor((640 - resizedWidth) / 2);
    const padY = Math.floor((640 - resizedHeight) / 2);
    const data = new Float32Array(3 * 640 * 640).fill(114 / 255);
    const planeSize = 640 * 640;

    for (let y = 0; y < resizedHeight; ++y) {
        for (let x = 0; x < resizedWidth; ++x) {
            const sourceOffset = (y * resizedWidth + x) * 3;
            const targetOffset = (y + padY) * 640 + x + padX;
            for (let channel = 0; channel < 3; ++channel) {
                data[channel * planeSize + targetOffset] = resized.data[sourceOffset + channel] / 255;
            }
        }
    }

    return {
        pixel_values: new Tensor('float32', data, [1, 3, 640, 640]),
        image_transform: {
            scale,
            padX,
            padY,
            originalWidth: image.width,
            originalHeight: image.height,
        },
    };
}

function intersectionOverUnion(a: Detection['box'], b: Detection['box']): number {
    const xmin = Math.max(a[0], b[0]);
    const ymin = Math.max(a[1], b[1]);
    const xmax = Math.min(a[2], b[2]);
    const ymax = Math.min(a[3], b[3]);
    const intersection = Math.max(0, xmax - xmin) * Math.max(0, ymax - ymin);
    const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
    const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
    return intersection / Math.max(areaA + areaB - intersection, Number.EPSILON);
}

export function postProcessYolo(
    output: YoloOutput,
    threshold = 0.5,
    targetSizes: number[][] | null = null,
    nmsThreshold = 0.7,
): Array<{ boxes: number[][]; scores: number[]; classes: number[] }> {
    const predictions = output.predictions;
    const [batch, channels, candidates] = predictions.dims;
    if (batch !== 1 || channels < 5) throw new Error(`Unexpected YOLO output shape: [${predictions.dims}].`);
    const data = predictions.data as Float32Array;
    const detections: Detection[] = [];
    for (let candidate = 0; candidate < candidates; ++candidate) {
        let score = Number.NEGATIVE_INFINITY;
        let classId = -1;
        for (let channel = 4; channel < channels; ++channel) {
            const candidateScore = data[channel * candidates + candidate];
            if (candidateScore > score) {
                score = candidateScore;
                classId = channel - 4;
            }
        }
        if (score < threshold) continue;
        const cx = data[candidate];
        const cy = data[candidates + candidate];
        const width = data[2 * candidates + candidate];
        const height = data[3 * candidates + candidate];
        detections.push({
            box: [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            score,
            classId,
        });
    }

    detections.sort((a, b) => b.score - a.score);
    const selected: Detection[] = [];
    for (const detection of detections) {
        if (
            selected.some(
                (other) =>
                    other.classId === detection.classId &&
                    intersectionOverUnion(other.box, detection.box) > nmsThreshold,
            )
        ) {
            continue;
        }
        selected.push(detection);
        if (selected.length === 300) break;
    }

    const transform = output.image_transform;
    const [targetHeight, targetWidth] = targetSizes?.[0] ?? [1, 1];
    const outputWidth = transform?.originalWidth ?? targetWidth;
    const outputHeight = transform?.originalHeight ?? targetHeight;
    const mapCoordinate = (value: number, padding: number, scale: number, target: number, original: number) =>
        Math.max(0, Math.min(target, ((value - padding) / scale) * (target / original)));
    return [
        {
            boxes: selected.map(({ box }) => {
                if (!transform) {
                    const scaleX = targetSizes ? targetWidth / 640 : 1 / 640;
                    const scaleY = targetSizes ? targetHeight / 640 : 1 / 640;
                    return [
                        Math.max(0, Math.min(targetWidth, box[0] * scaleX)),
                        Math.max(0, Math.min(targetHeight, box[1] * scaleY)),
                        Math.max(0, Math.min(targetWidth, box[2] * scaleX)),
                        Math.max(0, Math.min(targetHeight, box[3] * scaleY)),
                    ];
                }
                return [
                    mapCoordinate(box[0], transform.padX, transform.scale, targetWidth, outputWidth),
                    mapCoordinate(box[1], transform.padY, transform.scale, targetHeight, outputHeight),
                    mapCoordinate(box[2], transform.padX, transform.scale, targetWidth, outputWidth),
                    mapCoordinate(box[3], transform.padY, transform.scale, targetHeight, outputHeight),
                ];
            }),
            scores: selected.map(({ score }) => score),
            classes: selected.map(({ classId }) => classId),
        },
    ];
}

export function createYoloModel(model: CompiledModel, descriptor: LiteRtModelDescriptor): Function {
    const forward = async (inputs: YoloInputs): Promise<YoloOutput> => {
        const outputs = await runVisionModel(model, descriptor, { pixel_values: inputs.pixel_values });
        return { predictions: outputs.predictions, image_transform: inputs.image_transform };
    };
    return Object.assign(forward, {
        forward,
        capabilities: {
            forward: { version: 1 },
            objectDetection: { version: 1, preprocess: 'model', postprocess: 'model' },
        },
        preprocessObjectDetection: preprocessYolo,
        postProcessObjectDetection: postProcessYolo,
        dispose: () => model.delete(),
    });
}
