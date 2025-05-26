import type { TaskType, SentimentResult, NamedEntity, SummarizationResult, ClassificationResult } from './types';
export declare class AIManager {
    private state;
    private readonly configs;
    constructor();
    private init;
    private loadRemainingPipelines;
    private loadPipeline;
    analyzeSentiment(text: string): Promise<SentimentResult[]>;
    extractEntities(text: string): Promise<NamedEntity[]>;
    summarizeText(text: string): Promise<SummarizationResult>;
    classifyText(text: string): Promise<ClassificationResult[]>;
    answerQuestion(question: string, context: string): Promise<any>;
    generateEmbedding(text: string): Promise<number[]>;
    analyzeText(text: string, taskType: TaskType): Promise<any>;
    getStatus(): Record<string, string>;
    isReady(): boolean;
}
//# sourceMappingURL=ai-manager.d.ts.map