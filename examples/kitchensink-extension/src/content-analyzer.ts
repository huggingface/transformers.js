/**
 * ContentAnalyzer - Main content script for AI Context Copilot
 * Coordinates page analysis using modular services
 */

/// <reference types="chrome"/>

import { ExtensionMessage } from './types';
import { MessageHandler } from './services/message-handler';
import { PageMonitor } from './services/page-monitor';

export class ContentAnalyzer {
    private messageHandler: MessageHandler;
    private pageMonitor: PageMonitor;
    private isActive: boolean = false;

    constructor() {
        this.messageHandler = new MessageHandler();
        this.pageMonitor = new PageMonitor(() => this.handlePageChange());
        this.init();
    }

    private async init(): Promise<void> {
        // Setup message handling
        chrome.runtime.onMessage.addListener((message: ExtensionMessage, sender, sendResponse) => {
            this.messageHandler.handleMessage(message, sender, sendResponse);
            return true;
        });

        // Setup page monitoring and selection handling
        this.pageMonitor.setupPageMonitoring();
        this.pageMonitor.setupSelectionHandling((text: string) => this.analyzeText(text, 'text-classification'));

        // Initial page data extraction
        await this.messageHandler.extractPageData();
        this.isActive = true;

        console.log('AI Context Copilot: Content script initialized');
    }

    private handlePageChange(): void {
        if (this.isActive) {
            this.messageHandler.extractPageData();
        }
    }

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
     * Cleanup resources when content script is destroyed
     */
    destroy(): void {
        this.isActive = false;
        this.pageMonitor.cleanup();
    }
}

// Initialize content analyzer
new ContentAnalyzer();
