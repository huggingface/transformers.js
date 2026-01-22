import { ImageProcessor } from '../../image_processors_utils.js';

export class DPTImageProcessor extends ImageProcessor {}
export class DPTFeatureExtractor extends DPTImageProcessor {} // NOTE: extends DPTImageProcessor
