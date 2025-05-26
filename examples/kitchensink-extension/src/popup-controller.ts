/**
 * PopupController - Main controller for the popup interface
 * Coordinates between UI manager and analyzer components
 */

/// <reference types="chrome"/>

import { ExtensionMessage, PipelineStatusData } from './types';
import { PopupUIManager } from './popup-ui-manager';
import { PopupAnalyzer } from './popup-analyzer';

export class PopupController {
    private uiManager: PopupUIManager;
    private analyzer: PopupAnalyzer;

    constructor() {
        this.uiManager = new PopupUIManager();
        this.analyzer = new PopupAnalyzer();
        this.attachEventListeners();
        this.checkPipelineStatus();
    }

    private attachEventListeners(): void {
        const elements = this.uiManager.getElements();

        elements.openSidepanel.addEventListener('click', () => {
            chrome.sidePanel.open({ windowId: chrome.windows.WINDOW_ID_CURRENT });
        });

        elements.analyzeSentiment.addEventListener('click', () => {
            this.handleAnalyzeText(['text-classification']);
        });

        elements.extractEntities.addEventListener('click', () => {
            this.handleAnalyzeText(['token-classification']);
        });

        elements.summarizeText.addEventListener('click', () => {
            this.handleAnalyzeText(['summarization']);
        });

        elements.analyzePage.addEventListener('click', () => {
            this.handleAnalyzePage();
        });

        elements.extractMainContent.addEventListener('click', () => {
            this.handleExtractMainContent();
        });

        elements.customClassify.addEventListener('click', () => {
            this.handleCustomClassification();
        });

        elements.askQuestion.addEventListener('click', () => {
            this.handleQuestionAnswering();
        });

        elements.openOptions.addEventListener('click', (e) => {
            e.preventDefault();
            chrome.runtime.openOptionsPage();
        });

        elements.showHelp.addEventListener('click', (e) => {
            e.preventDefault();
            this.uiManager.showHelp();
        });

        elements.textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.handleAnalyzeText(['text-classification', 'token-classification']);
            }
        });
    }

    public async checkPipelineStatus(): Promise<void> {
        try {
            const response = await chrome.runtime.sendMessage({
                action: 'get-pipeline-status'
            } as ExtensionMessage);

            if (response.success) {
                const { loaded, loading } = response.data as PipelineStatusData;
                this.uiManager.updatePipelineStatus(loaded, loading);
            }
        } catch (error) {
            console.error('Error checking pipeline status:', error);
            this.uiManager.updatePipelineStatus([], [], 'Error connecting to AI service');
        }
    }

    private async handleAnalyzeText(tasks: string[]): Promise<void> {
        const text = this.uiManager.getTextInputValue();
        if (!text) {
            this.uiManager.showError('Please enter some text to analyze');
            return;
        }

        this.uiManager.showLoading('Analyzing text...');
        try {
            const result = await this.analyzer.analyzeText(text, tasks);
            this.uiManager.displayResults(result);
        } catch (error) {
            console.error('Text analysis error:', error);
            this.uiManager.showError(error instanceof Error ? error.message : 'Failed to analyze text');
        }
    }

    private async handleAnalyzePage(): Promise<void> {
        this.uiManager.showLoading('Analyzing page...');
        try {
            const { data, title } = await this.analyzer.analyzeCurrentPage();
            this.uiManager.displayResults(data, title);
        } catch (error) {
            console.error('Page analysis error:', error);
            this.uiManager.showError(error instanceof Error ? error.message : 'Failed to analyze page');
        }
    }

    private async handleExtractMainContent(): Promise<void> {
        this.uiManager.showLoading('Extracting content...');
        try {
            const content = await this.analyzer.extractMainContent();
            this.uiManager.setTextInputValue(content);
            this.uiManager.hideResults();
        } catch (error) {
            console.error('Content extraction error:', error);
            this.uiManager.showError(error instanceof Error ? error.message : 'Failed to extract content');
        }
    }

    private handleCustomClassification(): void {
        const labels = prompt('Enter classification labels (comma-separated):\nExample: positive, negative, neutral');
        if (labels) {
            const labelArray = labels.split(',').map(l => l.trim()).filter(l => l);
            if (labelArray.length > 0) {
                this.analyzeWithCustomLabels(labelArray);
            }
        }
    }

    private async analyzeWithCustomLabels(labels: string[]): Promise<void> {
        const text = this.uiManager.getTextInputValue();
        if (!text) {
            this.uiManager.showError('Please enter some text to classify');
            return;
        }

        this.uiManager.showLoading('Classifying...');
        try {
            const result = await this.analyzer.analyzeWithCustomLabels(text, labels);
            this.uiManager.displayResults(result);
        } catch (error) {
            console.error('Custom classification error:', error);
            this.uiManager.showError(error instanceof Error ? error.message : 'Failed to classify text');
        }
    }

    private handleQuestionAnswering(): void {
        const question = prompt('Ask a question about the text:');
        if (question) {
            this.answerQuestion(question);
        }
    }

    private async answerQuestion(question: string): Promise<void> {
        const text = this.uiManager.getTextInputValue();
        if (!text) {
            this.uiManager.showError('Please enter some text to ask questions about');
            return;
        }

        this.uiManager.showLoading('Finding answer...');
        try {
            const result = await this.analyzer.answerQuestion(text, question);
            this.uiManager.displayResults(result);
        } catch (error) {
            console.error('Question answering error:', error);
            this.uiManager.showError(error instanceof Error ? error.message : 'Failed to answer question');
        }
    }
}
