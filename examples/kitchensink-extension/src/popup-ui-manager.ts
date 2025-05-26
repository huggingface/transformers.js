/**
 * PopupUIManager - Handles UI element management and status updates for the popup
 */

import { AnalysisResult } from './types';
import { PopupResultRenderer } from './popup-result-renderer';

interface PopupElements {
    pipelineStatus: HTMLElement;
    textInput: HTMLTextAreaElement;
    resultsSection: HTMLElement;
    resultsContent: HTMLElement;
    openSidepanel: HTMLButtonElement;
    analyzeSentiment: HTMLButtonElement;
    extractEntities: HTMLButtonElement;
    summarizeText: HTMLButtonElement;
    analyzePage: HTMLButtonElement;
    extractMainContent: HTMLButtonElement;
    customClassify: HTMLButtonElement;
    askQuestion: HTMLButtonElement;
    openOptions: HTMLButtonElement;
    showHelp: HTMLButtonElement;
}

export class PopupUIManager {
    private elements!: PopupElements;
    private resultRenderer: PopupResultRenderer;

    constructor() {
        this.initializeElements();
        this.resultRenderer = new PopupResultRenderer(this.elements.resultsContent);
    }

    public initializeElements(): void {
        this.elements = {
            pipelineStatus: document.getElementById('pipeline-status')!,
            textInput: document.getElementById('text-input') as HTMLTextAreaElement,
            resultsSection: document.getElementById('results-section')!,
            resultsContent: document.getElementById('results-content')!,
            openSidepanel: document.getElementById('open-sidepanel') as HTMLButtonElement,
            analyzeSentiment: document.getElementById('analyze-sentiment') as HTMLButtonElement,
            extractEntities: document.getElementById('extract-entities') as HTMLButtonElement,
            summarizeText: document.getElementById('summarize-text') as HTMLButtonElement,
            analyzePage: document.getElementById('analyze-page') as HTMLButtonElement,
            extractMainContent: document.getElementById('extract-main-content') as HTMLButtonElement,
            customClassify: document.getElementById('custom-classify') as HTMLButtonElement,
            askQuestion: document.getElementById('ask-question') as HTMLButtonElement,
            openOptions: document.getElementById('open-options') as HTMLButtonElement,
            showHelp: document.getElementById('show-help') as HTMLButtonElement
        };
    }

    public getElements(): PopupElements {
        return this.elements;
    }

    public updatePipelineStatus(loaded: string[], loading: string[], error: string | null = null): void {
        const statusDot = this.elements.pipelineStatus.querySelector('.status-dot') as HTMLElement;
        const statusText = this.elements.pipelineStatus.querySelector('span:last-child') as HTMLElement;

        if (error) {
            statusDot.className = 'status-dot error';
            statusText.textContent = error;
            return;
        }

        if (loading.length > 0) {
            statusDot.className = 'status-dot loading';
            statusText.textContent = `Loading AI models... (${loading.length})`;
        } else if (loaded.length > 0) {
            statusDot.className = 'status-dot';
            statusText.textContent = `AI Ready (${loaded.length} models loaded)`;
        } else {
            statusDot.className = 'status-dot loading';
            statusText.textContent = 'AI models not loaded';
        }
    }

    public displayResults(data: AnalysisResult, title: string = 'Analysis Results'): void {
        this.elements.resultsSection.style.display = 'block';
        this.resultRenderer.renderResults(data, title);
        this.elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    public showLoading(message: string): void {
        this.elements.resultsSection.style.display = 'block';
        this.resultRenderer.renderLoading(message);
    }

    public showError(message: string): void {
        this.elements.resultsSection.style.display = 'block';
        this.resultRenderer.renderError(message);
    }

    public hideResults(): void {
        this.elements.resultsSection.style.display = 'none';
    }

    public showHelp(): void {
        this.elements.resultsSection.style.display = 'block';
        this.resultRenderer.renderHelp();
    }

    public setTextInputValue(value: string): void {
        this.elements.textInput.value = value;
    }

    public getTextInputValue(): string {
        return this.elements.textInput.value.trim();
    }
}
