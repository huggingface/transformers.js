/**
 * MessageHandler - Handles extension message routing and responses
 */

/// <reference types="chrome"/>

import { ExtensionMessage, PageContent } from '../types';
import { PageDataExtractor } from './page-data-extractor';
import { UIManager } from './ui-manager';

export class MessageHandler {
    private pageData: {
        text: string;
        images: any[];
        metadata: any;
        lastAnalyzed: number | null;
    };

    constructor() {
        this.pageData = {
            text: '',
            images: [],
            metadata: {},
            lastAnalyzed: null
        };
    }

    /**
     * Handle incoming messages from extension
     */
    async handleMessage(
        message: ExtensionMessage, 
        sender: chrome.runtime.MessageSender, 
        sendResponse: (response: any) => void
    ): Promise<void> {
        try {
            switch (message.action) {
                case 'extract-page-content':
                    await this.handleExtractPageContent(sendResponse);
                    break;

                case 'analyze-selection':
                    await this.handleAnalyzeSelection(message, sendResponse);
                    break;

                case 'analyze-element':
                    await this.handleAnalyzeElement(message, sendResponse);
                    break;

                case 'inject-analysis':
                    this.handleInjectAnalysis(message, sendResponse);
                    break;

                case 'highlight-entities':
                    this.handleHighlightEntities(message, sendResponse);
                    break;

                case 'summarize-page':
                    await this.handleSummarizePage(sendResponse);
                    break;

                default:
                    sendResponse({ success: false, error: 'Unknown action' });
            }
        } catch (error) {
            console.error('Content script error:', error);
            sendResponse({ success: false, error: (error as Error).message });
        }
    }

    /**
     * Extract and return page content
     */
    private async handleExtractPageContent(sendResponse: (response: any) => void): Promise<void> {
        await this.extractPageData();
        
        const pageContent: PageContent = {
            title: this.pageData.metadata.title,
            mainContent: this.pageData.text,
            url: this.pageData.metadata.url,
            images: this.pageData.images.map((img: any) => img.src)
        };

        sendResponse({ success: true, data: pageContent });
    }

    /**
     * Analyze selected text
     */
    private async handleAnalyzeSelection(message: ExtensionMessage, sendResponse: (response: any) => void): Promise<void> {
        const selection = PageDataExtractor.getSelectedText();
        
        if (selection) {
            const result = await this.analyzeText(selection, message.options?.tasks?.[0] || 'text-classification');
            sendResponse({ success: true, data: result });
        } else {
            sendResponse({ success: false, error: 'No text selected' });
        }
    }

    /**
     * Analyze element at coordinates
     */
    private async handleAnalyzeElement(message: ExtensionMessage, sendResponse: (response: any) => void): Promise<void> {
        const element = document.elementFromPoint(message.options?.x || 0, message.options?.y || 0);
        const result = await this.analyzeElement(element, message.options?.tasks?.[0] || 'text-classification');
        sendResponse({ success: true, data: result });
    }

    /**
     * Inject analysis UI
     */
    private handleInjectAnalysis(message: ExtensionMessage, sendResponse: (response: any) => void): void {
        UIManager.injectAnalysisUI(message.data);
        sendResponse({ success: true });
    }

    /**
     * Highlight named entities
     */
    private handleHighlightEntities(message: ExtensionMessage, sendResponse: (response: any) => void): void {
        UIManager.highlightNamedEntities(message.data?.entities || []);
        sendResponse({ success: true });
    }

    /**
     * Summarize page content
     */
    private async handleSummarizePage(sendResponse: (response: any) => void): Promise<void> {
        if (!this.pageData.text || this.pageData.text.length < 100) {
            sendResponse({ success: false, error: 'Not enough content to summarize' });
            return;
        }

        const summary = await this.analyzeText(this.pageData.text, 'summarization');
        sendResponse({ success: true, data: summary });
    }

    /**
     * Extract page data and update internal state
     */
    async extractPageData(): Promise<void> {
        this.pageData.text = PageDataExtractor.extractMainText();
        this.pageData.images = PageDataExtractor.extractImages();
        this.pageData.metadata = PageDataExtractor.extractMetadata(this.pageData.text);
        this.pageData.lastAnalyzed = Date.now();
    }

    /**
     * Send text analysis request to background script
     */
    private async analyzeText(text: string, taskType: string): Promise<any> {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage({
                action: 'analyzeText',
                text: text,
                taskType: taskType
            } as ExtensionMessage, (response) => {
                resolve(response);
            });
        });
    }

    /**
     * Analyze a specific DOM element
     */
    private async analyzeElement(element: Element | null, taskType: string): Promise<any> {
        if (!element) return null;

        let content = '';
        if (element.tagName === 'IMG') {
            content = (element as HTMLImageElement).src;
        } else {
            content = (element as HTMLElement).innerText || element.textContent || '';
        }

        return this.analyzeText(content, taskType);
    }
}
