/**
 * Background Service Worker for AI Context Copilot
 * Manages AI pipelines and handles cross-extension communication
 */

/// <reference types="chrome"/>

import type {
  ExtensionMessage,
  ExtensionResponse,
  TaskType
} from './types';
import { AIManager } from './ai-manager';

class BackgroundController {
  private aiManager: AIManager;

  constructor() {
    this.aiManager = new AIManager();
    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    // Extension installation
    chrome.runtime.onInstalled.addListener(this.handleInstalled.bind(this));

    // Message handling
    chrome.runtime.onMessage.addListener(this.handleMessage.bind(this));

    // Context menu setup
    this.setupContextMenus();

    console.log('🚀 Background controller initialized');
  }

  private handleInstalled(details: chrome.runtime.InstalledDetails): void {
    console.log('Extension installed:', details.reason);
    
    if (details.reason === 'install') {
      // Show welcome page or setup instructions
      chrome.tabs.create({
        url: 'https://github.com/xenova/transformers.js'
      });
    }
  }

  private handleMessage = (
    message: ExtensionMessage,
    sender: chrome.runtime.MessageSender,
    sendResponse: (response: ExtensionResponse) => void
  ): boolean => {
    console.log('📨 Received message:', message.action);

    switch (message.action) {
      case 'analyzeText':
        this.handleAnalyzeText(message, sendResponse);
        return true; // Keep channel open for async response

      case 'getPipelineStatus':
        this.handleGetPipelineStatus(sendResponse);
        return false;

      case 'processChat':
        this.handleProcessChat(message, sendResponse);
        return true;

      case 'customAnalysis':
        this.handleCustomAnalysis(message, sendResponse);
        return true;

      case 'pageDataUpdated':
        this.handlePageDataUpdated(message);
        return false;

      default:
        sendResponse({ success: false, error: `Unknown action: ${message.action}` });
        return false;
    }
  };

  private async handleAnalyzeText(
    message: ExtensionMessage,
    sendResponse: (response: ExtensionResponse) => void
  ): Promise<void> {
    try {
      const { text, taskType } = message;
      
      if (!this.aiManager.isReady()) {
        sendResponse({ success: false, error: 'AI Manager not ready yet' });
        return;
      }

      const result = await this.aiManager.analyzeText(text, taskType as TaskType);
      sendResponse({ success: true, data: result });

    } catch (error) {
      console.error('Analysis error:', error);
      sendResponse({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Analysis failed' 
      });
    }
  }

  private handleGetPipelineStatus(sendResponse: (response: ExtensionResponse) => void): void {
    const status = this.aiManager.getStatus();
    sendResponse({ success: true, data: status });
  }

  private async handleProcessChat(
    message: ExtensionMessage,
    sendResponse: (response: ExtensionResponse) => void
  ): Promise<void> {
    try {
      const { message: userMessage, context, history } = message;
      
      // Simple chat processing - in a real implementation, you'd use a proper chat model
      let response = "I'm analyzing your question...";
      
      if (context?.text) {
        // Try to answer based on page content
        if (userMessage.toLowerCase().includes('summary') || userMessage.toLowerCase().includes('summarize')) {
          const summary = await this.aiManager.summarizeText(context.text);
          response = `Here's a summary of the page: ${summary.summary_text}`;
        } else if (userMessage.toLowerCase().includes('sentiment')) {
          const sentiment = await this.aiManager.analyzeSentiment(context.text);
          const primarySentiment = sentiment[0];
          response = `The overall sentiment of this page is ${primarySentiment.label.toLowerCase()} with ${Math.round(primarySentiment.score * 100)}% confidence.`;
        } else if (userMessage.toLowerCase().includes('entities')) {
          const entities = await this.aiManager.extractEntities(context.text);
          const entityTypes = new Set(entities.map(e => e.entity_group));
          response = `I found ${entities.length} named entities on this page, including ${Array.from(entityTypes).join(', ')}.`;
        } else {
          // Try Q&A
          try {
            const answer = await this.aiManager.answerQuestion(userMessage, context.text);
            response = answer.answer || "I couldn't find a specific answer to your question in the page content.";
          } catch {
            response = "I can help you analyze this page. Try asking me to summarize it, analyze sentiment, or find entities!";
          }
        }
      } else {
        response = "I don't have access to the current page content. Please make sure you're on a page with text content.";
      }

      sendResponse({ success: true, data: response });

    } catch (error) {
      console.error('Chat processing error:', error);
      sendResponse({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Chat processing failed' 
      });
    }
  }

  private async handleCustomAnalysis(
    message: ExtensionMessage,
    sendResponse: (response: ExtensionResponse) => void
  ): Promise<void> {
    try {
      const { text, prompt } = message;
      
      // For custom analysis, we'll try to determine the best approach based on the prompt
      let taskType: TaskType = 'classification';
      
      if (prompt.toLowerCase().includes('sentiment') || prompt.toLowerCase().includes('emotion')) {
        taskType = 'sentiment';
      } else if (prompt.toLowerCase().includes('entities') || prompt.toLowerCase().includes('names')) {
        taskType = 'ner';
      } else if (prompt.toLowerCase().includes('summary') || prompt.toLowerCase().includes('summarize')) {
        taskType = 'summarization';
      }

      const result = await this.aiManager.analyzeText(text, taskType);
      sendResponse({ success: true, data: result });

    } catch (error) {
      console.error('Custom analysis error:', error);
      sendResponse({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Custom analysis failed' 
      });
    }
  }

  private handlePageDataUpdated(message: ExtensionMessage): void {
    console.log('📄 Page data updated:', message.data?.metadata?.title);
    // Could trigger automatic analysis or caching here
  }

  private async setupContextMenus(): Promise<void> {
    try {
      // Remove existing menus
      await chrome.contextMenus.removeAll();

      // Create parent menu
      chrome.contextMenus.create({
        id: 'ai-copilot-main',
        title: '🤖 AI Analyze',
        contexts: ['selection', 'page']
      });

      // Create sub-menus for different analysis types
      const menuItems = [
        { id: 'analyze-sentiment', title: '😊 Analyze Sentiment' },
        { id: 'extract-entities', title: '🏷️ Extract Entities' },
        { id: 'summarize-text', title: '📝 Summarize' },
        { id: 'classify-text', title: '📊 Classify' }
      ];

      menuItems.forEach(item => {
        chrome.contextMenus.create({
          id: item.id,
          parentId: 'ai-copilot-main',
          title: item.title,
          contexts: ['selection']
        });
      });

      // Handle context menu clicks
      chrome.contextMenus.onClicked.addListener(this.handleContextMenuClick.bind(this));

    } catch (error) {
      console.error('Failed to setup context menus:', error);
    }
  }

  private async handleContextMenuClick(
    info: chrome.contextMenus.OnClickData,
    tab?: chrome.tabs.Tab
  ): Promise<void> {
    if (!tab?.id) return;

    const taskTypeMap: Record<string, TaskType> = {
      'analyze-sentiment': 'sentiment',
      'extract-entities': 'ner',
      'summarize-text': 'summarization',
      'classify-text': 'classification'
    };

    const taskType = taskTypeMap[info.menuItemId as string];
    if (!taskType) return;

    try {
      if (info.selectionText) {
        // Analyze selected text
        const result = await this.aiManager.analyzeText(info.selectionText, taskType);
        
        // Send result to content script for display
        await chrome.tabs.sendMessage(tab.id, {
          action: 'injectAnalysis',
          data: result
        });
      }
    } catch (error) {
      console.error('Context menu analysis failed:', error);
    }
  }
}

// Initialize background controller
new BackgroundController();
