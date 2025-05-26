/**
 * AI Pipeline and Analysis Types
 */

export type PipelineType = 
  | 'text-classification'
  | 'token-classification' 
  | 'summarization'
  | 'zero-shot-classification'
  | 'question-answering'
  | 'feature-extraction'
  | 'sentiment-analysis';

export type TaskType = 
  | 'sentiment'
  | 'ner'
  | 'summarization'
  | 'classification'
  | 'translation'
  | 'qa'
  | 'embedding'
  | 'imageClassification'
  | 'objectDetection';

export type PipelineStatus = 'loading' | 'loaded' | 'error';

export interface AIAnalysisOptions {
  tasks: PipelineType[];
  labels?: string[];
  question?: string;
  title?: string;
}

export interface PipelineStatusData {
  loaded: PipelineType[];
  loading: PipelineType[];
}

export interface AIManagerState {
  pipelines: Map<string, any>;
  status: Map<string, 'loading' | 'loaded' | 'error'>;
  cache: Map<string, any>;
  isInitialized: boolean;
}

export interface PipelineConfig {
  name: string;
  task: PipelineType;
  model: string;
  quantized: boolean;
  status: PipelineStatus;
  lastUsed?: number;
  tokenizer?: string;
  revision?: string;
  cache_dir?: string;
  local_files_only?: boolean;
  device?: string;
}

export interface PipelineResult {
  [key: string]: any;
}
