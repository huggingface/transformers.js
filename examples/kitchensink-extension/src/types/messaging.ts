/**
 * Extension Messaging Types
 */

export interface ExtensionMessage {
  action: string;
  [key: string]: any;
}

export interface ExtensionResponse {
  success: boolean;
  data?: any;
  error?: string;
}

export interface AnalyzeTextMessage extends ExtensionMessage {
  action: 'analyzeText';
  text: string;
  taskType: string;
}

export interface GetPageDataMessage extends ExtensionMessage {
  action: 'getPageData';
}

export interface ProcessChatMessage extends ExtensionMessage {
  action: 'processChat';
  message: string;
  context: any;
  history: any[];
}

export interface CustomAnalysisMessage extends ExtensionMessage {
  action: 'customAnalysis';
  text: string;
  prompt: string;
}

export interface GetPipelineStatusMessage extends ExtensionMessage {
  action: 'getPipelineStatus';
}

export interface HighlightEntitiesMessage extends ExtensionMessage {
  action: 'highlightEntities';
  entities: any[];
}

export interface InjectAnalysisMessage extends ExtensionMessage {
  action: 'injectAnalysis';
  data: any;
}

export type MessageHandler = (
  message: ExtensionMessage,
  sender: any,
  sendResponse: (response: ExtensionResponse) => void
) => boolean | void;

export type TabMessageHandler = (
  message: ExtensionMessage,
  sender: any,
  sendResponse: (response: ExtensionResponse) => void
) => boolean | void;
