/**
 * UI Component Types
 */

import { PageData } from './content';
import { AnalysisResult } from './analysis';
import { ChatMessage } from './chat';
import { MemoryStats } from './memory';

export interface SidePanelState {
  currentTab: 'analysis' | 'conversation' | 'memory' | 'tools';
  pageData: PageData | null;
  analysisResults: Map<string, AnalysisResult>;
  chatHistory: ChatMessage[];
  memory: MemoryStats;
}

export interface ExportData {
  pageData: PageData | null;
  analysisResults: Record<string, any>;
  memory: MemoryStats;
  chatHistory: ChatMessage[];
  exported: string;
}

export type ExportFormat = 'json' | 'markdown' | 'csv';
