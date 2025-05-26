// ai-manager.ts
import { pipeline, Pipeline, env } from '@xenova/transformers';
import type {
  PipelineConfig,
  PipelineResult,
  ExtensionMessage,
  ExtensionResponse,
  MessageHandler,
  PageData,
  ChatMessage,
  AIManagerState,
  TaskType,
  SentimentResult,
  NamedEntity,
  SummarizationResult,
  ClassificationResult,
  AnalysisResult,
  QuestionAnsweringResult
} from './types';

// Configure Transformers.js for extension environment
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('dist/');

export class AIManager {
  private state: AIManagerState;
  private readonly configs: Map<TaskType, PipelineConfig>;

  constructor() {
    this.state = {
      pipelines: new Map(),
      status: new Map(),
      cache: new Map(),
      isInitialized: false
    };

    // Define pipeline configurations
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

    this.init();
  }

  private async init(): Promise<void> {
    console.log('🤖 AI Context Copilot: Initializing AI Manager...');
    try {
      await this.loadPipeline('sentiment');
      await this.loadPipeline('ner');
      this.state.isInitialized = true;
      console.log('✅ AI Manager initialized successfully');
      this.loadRemainingPipelines();
    } catch (error) {
      console.error('❌ Failed to initialize AI Manager:', error);
    }
  }

  private async loadRemainingPipelines(): Promise<void> {
    const remaining: TaskType[] = ['summarization', 'classification', 'qa', 'embedding'];
    for (const taskType of remaining) {
      try {
        await this.loadPipeline(taskType);
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.warn(`Failed to load ${taskType} pipeline:`, error);
      }
    }
  }

  private async loadPipeline(taskType: TaskType): Promise<any> {
    const config = this.configs.get(taskType);
    if (!config) throw new Error(`No configuration found for task: ${taskType}`);
    const existingPipeline = this.state.pipelines.get(taskType);
    if (existingPipeline) return existingPipeline;
    console.log(`📥 Loading ${taskType} pipeline...`);
    this.state.status.set(taskType, 'loading');
    try {
      const pipelineInstance = await pipeline(
        config.task,
        config.model,
        { quantized: config.quantized }
      );
      this.state.pipelines.set(taskType, pipelineInstance);
      this.state.status.set(taskType, 'loaded');
      config.status = 'loaded';
      config.lastUsed = Date.now();
      console.log(`✅ ${taskType} pipeline loaded successfully`);
      return pipelineInstance;
    } catch (error) {
      console.error(`❌ Failed to load ${taskType} pipeline:`, error);
      this.state.status.set(taskType, 'error');
      config.status = 'error';
      throw error;
    }
  }

  async analyzeSentiment(text: string): Promise<SentimentResult[]> {
    const pipelineInstance = await this.loadPipeline('sentiment');
    const result = await pipelineInstance(text);
    return Array.isArray(result) ? result : [result];
  }

  async extractEntities(text: string): Promise<NamedEntity[]> {
    const pipelineInstance = await this.loadPipeline('ner');
    const result = await pipelineInstance(text);
    return Array.isArray(result) ? result : [result];
  }

  async summarizeText(text: string): Promise<SummarizationResult> {
    const pipelineInstance = await this.loadPipeline('summarization');
    const maxLength = 1024;
    const inputText = text.length > maxLength ? text.substring(0, maxLength) : text;
    const result = await pipelineInstance(inputText, {
      max_length: 150,
      min_length: 30,
      do_sample: false
    });
    return Array.isArray(result) ? result[0] : result;
  }

  async classifyText(text: string): Promise<ClassificationResult[]> {
    const pipelineInstance = await this.loadPipeline('classification');
    const result = await pipelineInstance(text);
    return Array.isArray(result) ? result : [result];
  }

  async answerQuestion(question: string, context: string): Promise<any> {
    const pipelineInstance = await this.loadPipeline('qa');
    return await pipelineInstance(question, context);
  }

  async generateEmbedding(text: string): Promise<number[]> {
    const pipelineInstance = await this.loadPipeline('embedding');
    const result = await pipelineInstance(text, { pooling: 'mean', normalize: true });
    return result.data;
  }

  async analyzeText(text: string, taskType: TaskType): Promise<any> {
    if (!text || text.trim().length === 0) throw new Error('No text provided for analysis');
    const cacheKey = `${taskType}:${text.substring(0, 100)}`;
    const cached = this.state.cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < 300000) {
      return cached.data;
    }
    let result: any;
    switch (taskType) {
      case 'sentiment':
        result = await this.analyzeSentiment(text);
        break;
      case 'ner':
        result = await this.extractEntities(text);
        break;
      case 'summarization':
        result = await this.summarizeText(text);
        break;
      case 'classification':
        result = await this.classifyText(text);
        break;
      case 'qa':
        result = await this.answerQuestion("What is this about?", text);
        break;
      case 'embedding':
        result = await this.generateEmbedding(text);
        break;
      default:
        throw new Error(`Unsupported task type: ${taskType}`);
    }
    this.state.cache.set(cacheKey, {
      data: result,
      timestamp: Date.now()
    });
    return result;
  }

  getStatus(): Record<string, string> {
    const status: Record<string, string> = {};
    for (const [taskType, pipelineStatus] of this.state.status) {
      status[taskType] = pipelineStatus;
    }
    return status;
  }

  isReady(): boolean {
    return this.state.isInitialized;
  }
}
