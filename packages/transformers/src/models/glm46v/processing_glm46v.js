import { Qwen2VLProcessor } from '../qwen2_vl/processing_qwen2_vl.js';

export class Glm46VProcessor extends Qwen2VLProcessor {
    static image_token = '<|image|>';
}
