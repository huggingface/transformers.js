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
        document.addEventListener('mouseup', (event) => {
            const selection = window.getSelection();
            if (selection && selection.toString().trim().length > 0) {
                this.showSelectionTooltip(event.clientX, event.clientY, selection.toString());
            }
        });
    }

    private async extractPageData(): Promise<void> {
        const now = Date.now();
        if (this.pageData.lastAnalyzed && (now - this.pageData.lastAnalyzed < 30000)) {
            return;
        }
        try {
            this.pageData.text = this.extractMainContent();
            this.pageData.images = this.extractImages();
            this.pageData.metadata = this.extractMetadata();
            this.pageData.lastAnalyzed = now;
            chrome.runtime.sendMessage({
                action: 'pageDataUpdated',
                data: this.pageData
            } as ExtensionMessage);
        } catch (error) {
            console.error('Error extracting page data:', error);
        }
    }

    private extractMainContent(): string {
        const contentSelectors = [
            'main',
            'article',
            '.content',
            '.main-content',
            '#content',
            '#main',
            '.post-content',
            '.entry-content'
        ];
        let mainElement: Element | null = null;
        for (const selector of contentSelectors) {
            mainElement = document.querySelector(selector);
            if (mainElement) break;
        }
        if (!mainElement) {
            mainElement = document.body;
        }
        let text = (mainElement as HTMLElement).innerText || mainElement.textContent || '';
        const unwantedSelectors = ['nav', 'header', 'footer', '.navigation', '.sidebar'];
        unwantedSelectors.forEach(selector => {
            const elements = mainElement!.querySelectorAll(selector);
            elements.forEach(el => {
                const elementText = (el as HTMLElement).innerText || '';
                text = text.replace(elementText, '');
            });
        });
        text = text.replace(/\s+/g, ' ').trim();
        return text.substring(0, 10000);
    }

    private extractImages(): ImageData[] {
        const images: ImageData[] = [];
        const imgElements = document.querySelectorAll('img');
        imgElements.forEach((img, index) => {
            if (img.src && img.width > 50 && img.height > 50) {
                images.push({
                    src: img.src,
                    alt: img.alt || '',
                    width: img.width,
                    height: img.height,
                    index: index
                });
            }
        });
        return images.slice(0, 10);
    }

    private extractMetadata(): PageMetadata {
        const metadata: PageMetadata = {
            title: document.title,
            url: window.location.href,
            domain: window.location.hostname,
            description: '',
            keywords: [],
            language: document.documentElement.lang || 'en',
            wordCount: this.pageData.text ? this.pageData.text.split(' ').length : 0
        };
        const descMeta = document.querySelector('meta[name="description"]') as HTMLMetaElement;
        if (descMeta) {
            metadata.description = descMeta.getAttribute('content') || '';
        }
        const keywordsMeta = document.querySelector('meta[name="keywords"]') as HTMLMetaElement;
        if (keywordsMeta) {
            metadata.keywords = keywordsMeta.getAttribute('content')?.split(',').map(k => k.trim()) || [];
        }
        return metadata;
    }

    private getSelectedText(): string {
        const selection = window.getSelection();
        return selection ? selection.toString().trim() : '';
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

    private showSelectionTooltip(x: number, y: number, text: string): void {
        const existingTooltip = document.getElementById('ai-copilot-tooltip');
        if (existingTooltip) {
            existingTooltip.remove();
        }
        if (text.length < 10) return;
        const tooltip = document.createElement('div');
        tooltip.id = 'ai-copilot-tooltip';
        tooltip.innerHTML = `
            <div style="
                position: fixed;
                top: ${y + 10}px;
                left: ${x + 10}px;
                background: #2d3748;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                font-family: system-ui, -apple-system, sans-serif;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                z-index: 10000;
                max-width: 200px;
                cursor: pointer;
                border: 1px solid #4a5568;
            ">
                <div style="margin-bottom: 4px; font-weight: bold;">🤖 AI Analyze</div>
                <div style="font-size: 10px; color: #a0aec0;">Click to analyze selected text</div>
            </div>
        `;
        document.body.appendChild(tooltip);
        tooltip.addEventListener('click', async () => {
            tooltip.style.background = '#4a5568';
            tooltip.innerHTML = `
                <div style="padding: 8px 12px;">
                    <div>🔄 Analyzing...</div>
                </div>
            `;
            try {
                const result = await this.analyzeText(text, 'text-classification');
                if (result.success && result.data && Array.isArray(result.data) && result.data.length > 0) {
                    const sentiment = result.data[0] as AnalysisResult;
                    const emoji = sentiment.label === 'POSITIVE' ? '😊' : sentiment.label === 'NEGATIVE' ? '😞' : '😐';
                    tooltip.innerHTML = `
                        <div style="padding: 8px 12px;">
                            <div>${emoji} ${sentiment.label}</div>
                            <div style="font-size: 10px; color: #a0aec0;">
                                ${Math.round(sentiment.score * 100)}% confidence
                            </div>
                        </div>
                    `;
                }
            } catch (error) {
                tooltip.innerHTML = `
                    <div style="padding: 8px 12px; color: #fc8181;">
                        ❌ Analysis failed
                    </div>
                `;
            }
        });
        setTimeout(() => {
            if (tooltip.parentNode) {
                tooltip.remove();
            }
        }, 5000);
    }

    private highlightNamedEntities(entities: Entity[]): void {
        if (!entities || entities.length === 0) return;
        const colors: Record<string, string> = {
            'PER': '#ffd6cc',
            'ORG': '#cce5ff',
            'LOC': '#ccffcc',
            'MISC': '#ffffcc'
        };
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT
        );
        const textNodes: Text[] = [];
        let node: Node | null;
        while (node = walker.nextNode()) {
            textNodes.push(node as Text);
        }
        entities.forEach(entity => {
            const entityText = entity.word || entity.entity_group;
            const color = colors[entity.entity_group] || colors['MISC'];
            textNodes.forEach(textNode => {
                const text = textNode.textContent || '';
                const index = text.indexOf(entityText);
                if (index !== -1) {
                    const parent = textNode.parentNode;
                    if (!parent) return;
                    const before = text.substring(0, index);
                    const entitySpan = document.createElement('span');
                    const after = text.substring(index + entityText.length);
                    entitySpan.textContent = entityText;
                    entitySpan.style.backgroundColor = color;
                    entitySpan.style.padding = '1px 2px';
                    entitySpan.style.borderRadius = '2px';
                    entitySpan.title = `${entity.entity_group}: ${Math.round(entity.score * 100)}% confidence`;
                    const beforeNode = document.createTextNode(before);
                    const afterNode = document.createTextNode(after);
                    parent.insertBefore(beforeNode, textNode);
                    parent.insertBefore(entitySpan, textNode);
                    parent.insertBefore(afterNode, textNode);
                    parent.removeChild(textNode);
                }
            });
        });
    }

    private async summarizePage(): Promise<any> {
        if (!this.pageData.text || this.pageData.text.length < 100) {
            return { error: 'Not enough content to summarize' };
        }
        return this.analyzeText(this.pageData.text, 'summarization');
    }

    private injectAnalysisUI(data: any): void {
        const existingPanel = document.getElementById('ai-copilot-panel');
        if (existingPanel) {
            existingPanel.remove();
        }
        const panel = document.createElement('div');
        panel.id = 'ai-copilot-panel';
        panel.innerHTML = `
            <div style="
                position: fixed;
                top: 20px;
                right: 20px;
                width: 300px;
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                z-index: 10000;
                font-family: system-ui, -apple-system, sans-serif;
                max-height: 400px;
                overflow-y: auto;
            ">
                <div style="
                    padding: 12px 16px;
                    border-bottom: 1px solid #e2e8f0;
                    background: #f7fafc;
                    border-radius: 8px 8px 0 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div style="font-weight: bold; color: #2d3748;">🤖 AI Analysis</div>
                    <button onclick="this.closest('#ai-copilot-panel').remove()" style="
                        border: none;
                        background: none;
                        font-size: 18px;
                        cursor: pointer;
                        color: #718096;
                    ">×</button>
                </div>
                <div style="padding: 16px;">
                    ${this.formatAnalysisData(data)}
                </div>
            </div>
        `;
        document.body.appendChild(panel);
        setTimeout(() => {
            if (panel.parentNode) {
                panel.remove();
            }
        }, 30000);
    }

    private formatAnalysisData(data: any): string {
        if (!data) return '<div>No analysis data available</div>';
        let html = '';
        if (Array.isArray(data)) {
            html += '<div style="margin-bottom: 12px;">';
            data.forEach((item: AnalysisResult) => {
                const percentage = Math.round(item.score * 100);
                html += `
                    <div style="
                        display: flex; 
                        justify-content: space-between; 
                        margin-bottom: 6px;
                        padding: 6px 8px;
                        background: #f7fafc;
                        border-radius: 4px;
                    ">
                        <span style="font-weight: 500;">${item.label}</span>
                        <span style="color: #4a5568;">${percentage}%</span>
                    </div>
                `;
            });
            html += '</div>';
        } else if (typeof data === 'string') {
            html += `<div style="line-height: 1.5; color: #2d3748;">${data}</div>`;
        } else if (data.summary_text) {
            html += `<div style="line-height: 1.5; color: #2d3748;">${data.summary_text}</div>`;
        }
        return html;
    }
}
