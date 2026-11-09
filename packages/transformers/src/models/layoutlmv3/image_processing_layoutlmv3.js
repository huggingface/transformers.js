import { ImageProcessor } from '../../image_processors_utils.js';

/**
 * Image processor for LayoutLMv3. Images are resized, rescaled, and normalized
 * in the same way as other ViT-based document models.
 */
export class LayoutLMv3ImageProcessor extends ImageProcessor {}

export class LayoutLMv3FeatureExtractor extends LayoutLMv3ImageProcessor {}
