/**
 * SidePanelController for AI Context Copilot
 * Manages the main interface and coordinates with background script
 */

/// <reference types="chrome"/>

import { ExtensionMessage, AnalysisResult, PageContent, PipelineStatus } from './types';

interface PageData {
    text: string;
    images: string[];
    metadata: {
        title: string;
        url: string;
        wordCount: number;
        language: string;
    };
}

interface Memory {
    pagesAnalyzed: number;
    insightsGenerated: number;
    entitiesFound: number;
    timeline: MemoryEntry[];
}

interface MemoryEntry {
    type: string;
    content: string;
    timestamp: string;
    date: string;
}

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export class SidePanelController {
    // ... (class body is identical to the current SidePanelController in sidepanel.ts)
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new SidePanelController();
    });
} else {
    new SidePanelController();
}
