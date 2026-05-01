import type { DataType, DeviceType, ProgressCallback } from '@huggingface/transformers';
import { AutoModelForCausalLM, AutoTokenizer, ModelRegistry } from '@huggingface/transformers';
import type { ModelConfig } from './types.ts';

export class Model {
    readonly modelId: string;
    readonly device: DeviceType;
    readonly dtype: DataType | Record<string, DataType>;
    private _isInitialized = false;
    private _tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> | null = null;
    private _model: Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>> | null = null;

    get isInitialized(): boolean {
        return this._isInitialized;
    }

    get tokenizer(): Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> {
        if (this._tokenizer === null) {
            throw new Error('Model is not initialized. Call model.init() before accessing tokenizer.');
        }
        return this._tokenizer;
    }

    get model(): Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>> {
        if (this._model === null) {
            throw new Error('Model is not initialized. Call model.init() before accessing model.');
        }
        return this._model;
    }

    constructor(config: ModelConfig) {
        this.modelId = config.modelId;
        this.device = config.device ?? 'webgpu';
        this.dtype = config.dtype ?? 'q4f16';
    }

    async isCached(): Promise<boolean> {
        return await ModelRegistry.is_pipeline_cached('text-generation', this.modelId, {
            device: this.device,
            dtype: this.dtype,
        });
    }

    async downloadSize(): Promise<number> {
        const [totalSize, cachedSize] = await this.getCacheSizes();
        return Math.max(0, totalSize - cachedSize);
    }

    async cachedSize(): Promise<number> {
        const [, cachedSize] = await this.getCacheSizes();
        return cachedSize;
    }

    async init(progressCallback?: ProgressCallback): Promise<void> {
        if (this._isInitialized) {
            return;
        }

        const [tokenizer, model] = await Promise.all([
            AutoTokenizer.from_pretrained(this.modelId),
            AutoModelForCausalLM.from_pretrained(this.modelId, {
                device: this.device,
                dtype: this.dtype,
                progress_callback: progressCallback,
            }),
        ]);

        this._tokenizer = tokenizer;
        this._model = model;
        this._isInitialized = true;
    }

    static async load(config: ModelConfig, progressCallback?: ProgressCallback): Promise<Model> {
        const model = new Model(config);
        await model.init(progressCallback);
        return model;
    }

    private async getCacheSizes(): Promise<[number, number]> {
        const files: Array<string> = await ModelRegistry.get_pipeline_files('text-generation', this.modelId, {
            device: this.device,
            dtype: this.dtype,
        });

        let totalSize = 0;
        let cachedSize = 0;

        await Promise.all(
            files.map(async (file) => {
                const metadata = await ModelRegistry.get_file_metadata(this.modelId, file);
                const size = metadata.size ?? 0;
                totalSize += size;
                if (metadata.fromCache) {
                    cachedSize += size;
                }
            }),
        );

        return [totalSize, cachedSize];
    }
}
