import { Processor } from "../../base/processing_utils.js";
import { U2NetImageProcessor } from "./image_processing_u2net.js";

export class U2NetProcessor extends Processor {
    static image_processor_class = U2NetImageProcessor

    async _call(...args) {
        return await this.image_processor(...args);
    }
}
