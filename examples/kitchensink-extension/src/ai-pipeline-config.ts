/**
 * AIPipelineConfig - Manages AI pipeline configurations
 */

import type { PipelineConfig, TaskType } from './types';

export class AIPipelineConfig {
    private readonly configs: Map<TaskType, PipelineConfig>;

    constructor() {
        this.configs = new Map([
            ['sentiment', {
                name: 'sentiment-analysis',
                task: 'sentiment-analysis',
                model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
                quantized: true,
                status: 'loading'
            }],
            ['ner', {
                name: 'token-classification',
                task: 'token-classification',
                model: 'Xenova/distilbert-base-NER',
                quantized: true,
                status: 'loading'
            }],
            ['summarization', {
                name: 'summarization',
                task: 'summarization',
                model: 'Xenova/distilbart-cnn-6-6',
                quantized: true,
                status: 'loading'
            }],
            ['classification', {
                name: 'text-classification',
                task: 'text-classification',
                model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
                quantized: true,
                status: 'loading'
            }],
            ['qa', {
                name: 'question-answering',
                task: 'question-answering',
                model: 'Xenova/distilbert-base-cased-distilled-squad',
                quantized: true,
                status: 'loading'
            }],
            ['embedding', {
                name: 'feature-extraction',
                task: 'feature-extraction',
                model: 'Xenova/all-MiniLM-L6-v2',
                quantized: true,
                status: 'loading'
            }]
        ]);
    }

    public getConfig(taskType: TaskType): PipelineConfig | undefined {
        return this.configs.get(taskType);
    }

    public getAllConfigs(): Map<TaskType, PipelineConfig> {
        return new Map(this.configs);
    }

    public updateStatus(taskType: TaskType, status: 'loading' | 'loaded' | 'error'): void {
        const config = this.configs.get(taskType);
        if (config) {
            config.status = status;
            if (status === 'loaded') {
                config.lastUsed = Date.now();
            }
        }
    }

    public getCoreTaskTypes(): TaskType[] {
        return ['sentiment', 'ner'];
    }

    public getSecondaryTaskTypes(): TaskType[] {
        return ['summarization', 'classification', 'qa', 'embedding'];
    }
}
