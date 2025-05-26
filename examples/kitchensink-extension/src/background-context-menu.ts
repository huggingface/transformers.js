/**
 * BackgroundContextMenu - Handles context menu setup and interactions
 */

/// <reference types="chrome"/>

import type { TaskType } from './types';
import { AIManager } from './ai-manager';

export class BackgroundContextMenu {
    private aiManager: AIManager;

    constructor(aiManager: AIManager) {
        this.aiManager = aiManager;
        this.setupContextMenus();
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
