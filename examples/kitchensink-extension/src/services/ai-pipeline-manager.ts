// services/ai-pipeline-manager.ts
import { PipelineType, AIAnalysisOptions, AnalysisResult } from '../types';

export class AIPipelineManager {
    private pipelines: Map<PipelineType, any> = new Map();
    private loadingPromises: Map<PipelineType, Promise<any>> = new Map();
    private modelCache: Map<string, any> = new Map();

    async loadPipeline(type: PipelineType): Promise<any> {
        if (this.pipelines.has(type)) {
            return this.pipelines.get(type);
        }

        if (this.loadingPromises.has(type)) {
            return this.loadingPromises.get(type);
        }

        const loadPromise = this.createPipeline(type);
        this.loadingPromises.set(type, loadPromise);

        try {
            const pipeline = await loadPromise;
            this.pipelines.set(type, pipeline);
            this.loadingPromises.delete(type);
            return pipeline;
        } catch (error) {
            this.loadingPromises.delete(type);
            throw error;
        }
    }

    private async createPipeline(type: PipelineType): Promise<any> {
        const { pipeline } = await import('@xenova/transformers');
        
        const modelConfigs = {
            'text-classification': 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
            'token-classification': 'Xenova/bert-base-NER',
            'summarization': 'Xenova/distilbart-cnn-6-6',
            'zero-shot-classification': 'Xenova/mobilebert-uncased-mnli',
            'question-answering': 'Xenova/distilbert-base-cased-distilled-squad',
            'feature-extraction': 'Xenova/all-MiniLM-L6-v2'
        };

        const modelId = modelConfigs[type];
        if (!modelId) {
            throw new Error(`Unsupported pipeline type: ${type}`);
        }

        return await pipeline(type, modelId);
    }

    async analyzeText(text: string, options: AIAnalysisOptions): Promise<AnalysisResult> {
        const result: AnalysisResult = {};

        for (const task of options.tasks) {
            try {
                const pipeline = await this.loadPipeline(task);
                result[task] = await this.runAnalysis(pipeline, text, task, options);
            } catch (error) {
                result[`${task}_error`] = error instanceof Error ? error.message : String(error);
            }
        }

        return result;
    }

    private async runAnalysis(pipeline: any, text: string, task: PipelineType, options: AIAnalysisOptions): Promise<any> {
        switch (task) {
            case 'text-classification':
                return await pipeline(text);
            case 'token-classification':
                return await pipeline(text);
            case 'summarization':
                return await pipeline(text, { max_length: 100, min_length: 30 });
            case 'zero-shot-classification':
                if (!options.labels) throw new Error('Labels required for zero-shot classification');
                return await pipeline(text, options.labels);
            case 'question-answering':
                if (!options.question) throw new Error('Question required for Q&A');
                return await pipeline({ question: options.question, context: text });
            case 'feature-extraction':
                return await pipeline(text, { pooling: 'mean', normalize: true });
            default:
                throw new Error(`Unsupported task: ${task}`);
        }
    }

    getLoadedPipelines(): PipelineType[] {
        return Array.from(this.pipelines.keys());
    }

    getLoadingPipelines(): PipelineType[] {
        return Array.from(this.loadingPromises.keys());
    }

    dispose(): void {
        this.pipelines.clear();
        this.loadingPromises.clear();
        this.modelCache.clear();
    }
}
