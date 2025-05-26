/**
 * PopupAnalyzer - Handles all analysis operations for the popup
 */

/// <reference types="chrome"/>

import { AnalysisResult, ExtensionMessage, PageContent } from './types';

export class PopupAnalyzer {
    
    public async analyzeText(text: string, tasks: string[]): Promise<AnalysisResult> {
        const response = await chrome.runtime.sendMessage({
            action: 'analyze-text',
            text: text,
            options: { tasks: tasks }
        } as ExtensionMessage);

        if (!response.success) {
            throw new Error(response.error || 'Analysis failed');
        }

        return response.data as AnalysisResult;
    }

    public async analyzeCurrentPage(): Promise<{ data: AnalysisResult; title: string }> {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        const response = await chrome.tabs.sendMessage(tab.id!, {
            action: 'extract-page-content'
        } as ExtensionMessage);

        if (!response || !response.success) {
            throw new Error('Could not extract page content');
        }

        const pageContent = response.data as PageContent;
        
        const analysisResponse = await chrome.runtime.sendMessage({
            action: 'analyze-text',
            text: pageContent.mainContent,
            options: { 
                tasks: ['text-classification', 'summarization', 'token-classification'],
                title: pageContent.title
            }
        } as ExtensionMessage);

        if (!analysisResponse.success) {
            throw new Error(analysisResponse.error || 'Page analysis failed');
        }

        return {
            data: analysisResponse.data as AnalysisResult,
            title: 'Page Analysis'
        };
    }

    public async extractMainContent(): Promise<string> {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        const response = await chrome.tabs.sendMessage(tab.id!, {
            action: 'extract-page-content'
        } as ExtensionMessage);

        if (!response || !response.success) {
            throw new Error('Could not extract page content');
        }

        const pageContent = response.data as PageContent;
        const content = pageContent.mainContent.substring(0, 1000) + 
            (pageContent.mainContent.length > 1000 ? '...' : '');
        
        return content;
    }

    public async analyzeWithCustomLabels(text: string, labels: string[]): Promise<AnalysisResult> {
        const response = await chrome.runtime.sendMessage({
            action: 'analyze-text',
            text: text,
            options: { 
                tasks: ['zero-shot-classification'],
                labels: labels
            }
        } as ExtensionMessage);

        if (!response.success) {
            throw new Error(response.error || 'Classification failed');
        }

        return response.data as AnalysisResult;
    }

    public async answerQuestion(text: string, question: string): Promise<AnalysisResult> {
        const response = await chrome.runtime.sendMessage({
            action: 'analyze-text',
            text: text,
            options: { 
                tasks: ['question-answering'],
                question: question
            }
        } as ExtensionMessage);

        if (!response.success) {
            throw new Error(response.error || 'Question answering failed');
        }

        return response.data as AnalysisResult;
    }
}
