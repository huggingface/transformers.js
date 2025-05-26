/**
 * BackgroundMessageHandler - Handles all message processing for the background script
 */

/// <reference types="chrome"/>

import type { ExtensionMessage, ExtensionResponse, TaskType } from './types';
import { AIManager } from './ai-manager';

export class BackgroundMessageHandler {
    private aiManager: AIManager;

    constructor(aiManager: AIManager) {
        this.aiManager = aiManager;
    }

    public handleMessage = (
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
            
            let response = "I'm analyzing your question...";
            
            if (context?.text) {
                response = await this.processContextualChat(userMessage, context.text);
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

    private async processContextualChat(userMessage: string, contextText: string): Promise<string> {
        const message = userMessage.toLowerCase();
        
        if (message.includes('summary') || message.includes('summarize')) {
            const summary = await this.aiManager.summarizeText(contextText);
            return `Here's a summary of the page: ${summary.summary_text}`;
        } 
        
        if (message.includes('sentiment')) {
            const sentiment = await this.aiManager.analyzeSentiment(contextText);
            const primarySentiment = sentiment[0];
            return `The overall sentiment of this page is ${primarySentiment.label.toLowerCase()} with ${Math.round(primarySentiment.score * 100)}% confidence.`;
        } 
        
        if (message.includes('entities')) {
            const entities = await this.aiManager.extractEntities(contextText);
            const entityTypes = new Set(entities.map(e => e.entity_group));
            return `I found ${entities.length} named entities on this page, including ${Array.from(entityTypes).join(', ')}.`;
        }
        
        // Try Q&A
        try {
            const answer = await this.aiManager.answerQuestion(userMessage, contextText);
            return answer.answer || "I couldn't find a specific answer to your question in the page content.";
        } catch {
            return "I can help you analyze this page. Try asking me to summarize it, analyze sentiment, or find entities!";
        }
    }

    private async handleCustomAnalysis(
        message: ExtensionMessage,
        sendResponse: (response: ExtensionResponse) => void
    ): Promise<void> {
        try {
            const { text, prompt } = message;
            
            const taskType = this.determineTaskType(prompt);
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

    private determineTaskType(prompt: string): TaskType {
        const lowerPrompt = prompt.toLowerCase();
        
        if (lowerPrompt.includes('sentiment') || lowerPrompt.includes('emotion')) {
            return 'sentiment';
        }
        if (lowerPrompt.includes('entities') || lowerPrompt.includes('names')) {
            return 'ner';
        }
        if (lowerPrompt.includes('summary') || lowerPrompt.includes('summarize')) {
            return 'summarization';
        }
        
        return 'classification';
    }

    private handlePageDataUpdated(message: ExtensionMessage): void {
        console.log('📄 Page data updated:', message.data?.metadata?.title);
        // Could trigger automatic analysis or caching here
    }
}
