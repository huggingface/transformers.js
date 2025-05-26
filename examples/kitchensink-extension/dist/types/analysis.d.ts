/**
 * Analysis Result Types
 */
export interface AnalysisResult {
    [key: string]: any;
    sentiment?: SentimentResult[];
    entities?: NamedEntity[];
    summary?: SummarizationResult[];
    classification?: ClassificationResult;
    answer?: QuestionAnsweringResult;
    embedding?: number[];
}
export interface SentimentResult {
    label: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
    score: number;
}
export interface NamedEntity {
    entity_group: 'PER' | 'ORG' | 'LOC' | 'MISC';
    word: string;
    start: number;
    end: number;
    score: number;
}
export interface SummarizationResult {
    summary_text: string;
}
export interface ClassificationResult {
    label: string;
    score: number;
}
export interface TranslationResult {
    translation_text: string;
}
export interface QuestionAnsweringResult {
    answer: string;
    score: number;
    start: number;
    end: number;
}
//# sourceMappingURL=analysis.d.ts.map