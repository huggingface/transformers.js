import { PipelineType, AIAnalysisOptions, AnalysisResult } from '../types';
export declare class AIPipelineManager {
    private pipelines;
    private loadingPromises;
    private modelCache;
    loadPipeline(type: PipelineType): Promise<any>;
    private createPipeline;
    analyzeText(text: string, options: AIAnalysisOptions): Promise<AnalysisResult>;
    private runAnalysis;
    getLoadedPipelines(): PipelineType[];
    getLoadingPipelines(): PipelineType[];
    dispose(): void;
}
//# sourceMappingURL=ai-pipeline-manager.d.ts.map