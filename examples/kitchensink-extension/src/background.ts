/**
 * Background Service Worker for AI Context Copilot
 * Main controller that coordinates AI manager, message handling, and context menus
 */

/// <reference types="chrome"/>

import { AIManager } from './ai-manager';
import { BackgroundMessageHandler } from './background-message-handler';
import { BackgroundContextMenu } from './background-context-menu';

class BackgroundController {
    private aiManager: AIManager;
    private messageHandler: BackgroundMessageHandler;
    private contextMenu: BackgroundContextMenu;

    constructor() {
        this.aiManager = new AIManager();
        this.messageHandler = new BackgroundMessageHandler(this.aiManager);
        this.contextMenu = new BackgroundContextMenu(this.aiManager);
        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        // Extension installation
        chrome.runtime.onInstalled.addListener(this.handleInstalled.bind(this));

        // Message handling
        chrome.runtime.onMessage.addListener(this.messageHandler.handleMessage);

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
}

// Initialize background controller
new BackgroundController();
