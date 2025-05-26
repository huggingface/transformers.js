/**
 * PopupResultRenderer - Handles rendering of analysis results in the popup
 */

import { AnalysisResult } from './types';

export class PopupResultRenderer {
    private resultsContent: HTMLElement;

    constructor(resultsContent: HTMLElement) {
        this.resultsContent = resultsContent;
    }

    public renderResults(data: AnalysisResult, title: string = 'Analysis Results'): void {
        this.resultsContent.innerHTML = '';

        const titleEl = document.createElement('div');
        titleEl.className = 'result-label';
        titleEl.textContent = title;
        this.resultsContent.appendChild(titleEl);

        Object.entries(data).forEach(([key, value]) => {
            if (key.endsWith('_error')) {
                this.addResultItem(key.replace('_error', ''), `Error: ${value}`, 'error');
                return;
            }

            switch (key) {
                case 'sentiment':
                    this.addSentimentResult(value);
                    break;
                case 'entities':
                    this.addEntitiesResult(value);
                    break;
                case 'summary':
                    this.addSummaryResult(value);
                    break;
                case 'classification':
                    this.addClassificationResult(value);
                    break;
                case 'answer':
                    this.addAnswerResult(value);
                    break;
                default:
                    this.addGenericResult(key, value);
            }
        });
    }

    public renderLoading(message: string): void {
        this.resultsContent.innerHTML = `
            <div class="result-item">
                <div class="result-label">
                    <span class="spinner"></span> ${message}
                </div>
            </div>
        `;
    }

    public renderError(message: string): void {
        this.resultsContent.innerHTML = `
            <div class="result-item" style="background: #f8d7da; color: #721c24;">
                <div class="result-label">Error</div>
                <div class="result-value">${message}</div>
            </div>
        `;
    }

    public renderHelp(): void {
        this.resultsContent.innerHTML = `
            <div class="result-item">
                <div class="result-label">How to Use</div>
                <div class="result-value">
                    • Enter text and click analysis buttons<br>
                    • Right-click on any page to analyze content<br>
                    • Use Ctrl+Enter in text box for quick analysis<br>
                    • Open side panel for advanced features
                </div>
            </div>
        `;
    }

    private addSentimentResult(sentiment: any): void {
        if (Array.isArray(sentiment) && sentiment.length > 0) {
            const result = sentiment[0];
            this.addResultItem('Sentiment', result.label, 'sentiment', result.score);
        }
    }

    private addEntitiesResult(entities: any): void {
        if (Array.isArray(entities) && entities.length > 0) {
            const entityText = entities.map((e: any) => `${e.word} (${e.entity})`).join(', ');
            this.addResultItem('Entities', entityText, 'entities');
        } else {
            this.addResultItem('Entities', 'No entities found', 'entities');
        }
    }

    private addSummaryResult(summary: any): void {
        if (Array.isArray(summary) && summary.length > 0) {
            this.addResultItem('Summary', summary[0].summary_text, 'summary');
        }
    }

    private addClassificationResult(classification: any): void {
        if (classification && classification.labels) {
            const topResult = `${classification.labels[0]} (${(classification.scores[0] * 100).toFixed(1)}%)`;
            this.addResultItem('Classification', topResult, 'classification');
        }
    }

    private addAnswerResult(answer: any): void {
        if (answer && answer.answer) {
            this.addResultItem('Answer', answer.answer, 'answer', answer.score);
        }
    }

    private addGenericResult(key: string, value: any): void {
        const displayValue = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
        this.addResultItem(key, displayValue, 'generic');
    }

    private addResultItem(label: string, value: string, type: string = 'default', confidence: number | null = null): void {
        const item = document.createElement('div');
        item.className = 'result-item';

        const labelEl = document.createElement('div');
        labelEl.className = 'result-label';
        labelEl.textContent = label;

        const valueEl = document.createElement('div');
        valueEl.className = 'result-value';
        valueEl.textContent = value;

        item.appendChild(labelEl);
        item.appendChild(valueEl);

        if (confidence !== null && confidence !== undefined) {
            const confidenceEl = document.createElement('span');
            confidenceEl.className = 'confidence-score';
            confidenceEl.textContent = `${(confidence * 100).toFixed(1)}%`;
            valueEl.appendChild(confidenceEl);
        }

        this.resultsContent.appendChild(item);
    }
}
