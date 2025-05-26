// popup-controller.ts - Popup interface logic
/// <reference types="chrome"/>
class PopupController {
    elements;
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.checkPipelineStatus();
    }
    initializeElements() {
        this.elements = {
            pipelineStatus: document.getElementById('pipeline-status'),
            textInput: document.getElementById('text-input'),
            resultsSection: document.getElementById('results-section'),
            resultsContent: document.getElementById('results-content'),
            openSidepanel: document.getElementById('open-sidepanel'),
            analyzeSentiment: document.getElementById('analyze-sentiment'),
            extractEntities: document.getElementById('extract-entities'),
            summarizeText: document.getElementById('summarize-text'),
            analyzePage: document.getElementById('analyze-page'),
            extractMainContent: document.getElementById('extract-main-content'),
            customClassify: document.getElementById('custom-classify'),
            askQuestion: document.getElementById('ask-question'),
            openOptions: document.getElementById('open-options'),
            showHelp: document.getElementById('show-help')
        };
    }
    attachEventListeners() {
        this.elements.openSidepanel.addEventListener('click', () => {
            chrome.sidePanel.open({ windowId: chrome.windows.WINDOW_ID_CURRENT });
        });
        this.elements.analyzeSentiment.addEventListener('click', () => {
            this.analyzeText(['text-classification']);
        });
        this.elements.extractEntities.addEventListener('click', () => {
            this.analyzeText(['token-classification']);
        });
        this.elements.summarizeText.addEventListener('click', () => {
            this.analyzeText(['summarization']);
        });
        this.elements.analyzePage.addEventListener('click', () => {
            this.analyzeCurrentPage();
        });
        this.elements.extractMainContent.addEventListener('click', () => {
            this.extractMainContent();
        });
        this.elements.customClassify.addEventListener('click', () => {
            this.showCustomClassification();
        });
        this.elements.askQuestion.addEventListener('click', () => {
            this.showQuestionInterface();
        });
        this.elements.openOptions.addEventListener('click', (e) => {
            e.preventDefault();
            chrome.runtime.openOptionsPage();
        });
        this.elements.showHelp.addEventListener('click', (e) => {
            e.preventDefault();
            this.showHelp();
        });
        this.elements.textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.analyzeText(['text-classification', 'token-classification']);
            }
        });
    }
    async checkPipelineStatus() {
        try {
            const response = await chrome.runtime.sendMessage({
                action: 'get-pipeline-status'
            });
            if (response.success) {
                const { loaded, loading } = response.data;
                this.updatePipelineStatus(loaded, loading);
            }
        }
        catch (error) {
            console.error('Error checking pipeline status:', error);
            this.updatePipelineStatus([], [], 'Error connecting to AI service');
        }
    }
    updatePipelineStatus(loaded, loading, error = null) {
        const statusDot = this.elements.pipelineStatus.querySelector('.status-dot');
        const statusText = this.elements.pipelineStatus.querySelector('span:last-child');
        if (error) {
            statusDot.className = 'status-dot error';
            statusText.textContent = error;
            return;
        }
        if (loading.length > 0) {
            statusDot.className = 'status-dot loading';
            statusText.textContent = `Loading AI models... (${loading.length})`;
        }
        else if (loaded.length > 0) {
            statusDot.className = 'status-dot';
            statusText.textContent = `AI Ready (${loaded.length} models loaded)`;
        }
        else {
            statusDot.className = 'status-dot loading';
            statusText.textContent = 'AI models not loaded';
        }
    }
    async analyzeText(tasks) {
        const text = this.elements.textInput.value.trim();
        if (!text) {
            this.showError('Please enter some text to analyze');
            return;
        }
        this.showLoading('Analyzing text...');
        try {
            const response = await chrome.runtime.sendMessage({
                action: 'analyze-text',
                text: text,
                options: { tasks: tasks }
            });
            if (response.success) {
                this.displayResults(response.data);
            }
            else {
                this.showError(response.error || 'Analysis failed');
            }
        }
        catch (error) {
            console.error('Text analysis error:', error);
            this.showError('Failed to analyze text');
        }
    }
    async analyzeCurrentPage() {
        this.showLoading('Analyzing page...');
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const response = await chrome.tabs.sendMessage(tab.id, {
                action: 'extract-page-content'
            });
            if (response && response.success) {
                const pageContent = response.data;
                const analysisResponse = await chrome.runtime.sendMessage({
                    action: 'analyze-text',
                    text: pageContent.mainContent,
                    options: {
                        tasks: ['text-classification', 'summarization', 'token-classification'],
                        title: pageContent.title
                    }
                });
                if (analysisResponse.success) {
                    this.displayResults(analysisResponse.data, 'Page Analysis');
                }
                else {
                    this.showError(analysisResponse.error || 'Page analysis failed');
                }
            }
            else {
                this.showError('Could not extract page content');
            }
        }
        catch (error) {
            console.error('Page analysis error:', error);
            this.showError('Failed to analyze page');
        }
    }
    async extractMainContent() {
        this.showLoading('Extracting content...');
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const response = await chrome.tabs.sendMessage(tab.id, {
                action: 'extract-page-content'
            });
            if (response && response.success) {
                const pageContent = response.data;
                this.elements.textInput.value = pageContent.mainContent.substring(0, 1000) +
                    (pageContent.mainContent.length > 1000 ? '...' : '');
                this.hideResults();
            }
            else {
                this.showError('Could not extract page content');
            }
        }
        catch (error) {
            console.error('Content extraction error:', error);
            this.showError('Failed to extract content');
        }
    }
    showCustomClassification() {
        const labels = prompt('Enter classification labels (comma-separated):\nExample: positive, negative, neutral');
        if (labels) {
            const labelArray = labels.split(',').map(l => l.trim()).filter(l => l);
            if (labelArray.length > 0) {
                this.analyzeWithCustomLabels(labelArray);
            }
        }
    }
    async analyzeWithCustomLabels(labels) {
        const text = this.elements.textInput.value.trim();
        if (!text) {
            this.showError('Please enter some text to classify');
            return;
        }
        this.showLoading('Classifying...');
        try {
            const response = await chrome.runtime.sendMessage({
                action: 'analyze-text',
                text: text,
                options: {
                    tasks: ['zero-shot-classification'],
                    labels: labels
                }
            });
            if (response.success) {
                this.displayResults(response.data);
            }
            else {
                this.showError(response.error || 'Classification failed');
            }
        }
        catch (error) {
            console.error('Custom classification error:', error);
            this.showError('Failed to classify text');
        }
    }
    showQuestionInterface() {
        const question = prompt('Ask a question about the text:');
        if (question) {
            this.answerQuestion(question);
        }
    }
    async answerQuestion(question) {
        const text = this.elements.textInput.value.trim();
        if (!text) {
            this.showError('Please enter some text to ask questions about');
            return;
        }
        this.showLoading('Finding answer...');
        try {
            const response = await chrome.runtime.sendMessage({
                action: 'analyze-text',
                text: text,
                options: {
                    tasks: ['question-answering'],
                    question: question
                }
            });
            if (response.success) {
                this.displayResults(response.data);
            }
            else {
                this.showError(response.error || 'Question answering failed');
            }
        }
        catch (error) {
            console.error('Question answering error:', error);
            this.showError('Failed to answer question');
        }
    }
    displayResults(data, title = 'Analysis Results') {
        this.elements.resultsContent.innerHTML = '';
        this.elements.resultsSection.style.display = 'block';
        const titleEl = document.createElement('div');
        titleEl.className = 'result-label';
        titleEl.textContent = title;
        this.elements.resultsContent.appendChild(titleEl);
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
        this.elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    addSentimentResult(sentiment) {
        if (Array.isArray(sentiment) && sentiment.length > 0) {
            const result = sentiment[0];
            this.addResultItem('Sentiment', result.label, 'sentiment', result.score);
        }
    }
    addEntitiesResult(entities) {
        if (Array.isArray(entities) && entities.length > 0) {
            const entityText = entities.map((e) => `${e.word} (${e.entity})`).join(', ');
            this.addResultItem('Entities', entityText, 'entities');
        }
        else {
            this.addResultItem('Entities', 'No entities found', 'entities');
        }
    }
    addSummaryResult(summary) {
        if (Array.isArray(summary) && summary.length > 0) {
            this.addResultItem('Summary', summary[0].summary_text, 'summary');
        }
    }
    addClassificationResult(classification) {
        if (classification && classification.labels) {
            const topResult = `${classification.labels[0]} (${(classification.scores[0] * 100).toFixed(1)}%)`;
            this.addResultItem('Classification', topResult, 'classification');
        }
    }
    addAnswerResult(answer) {
        if (answer && answer.answer) {
            this.addResultItem('Answer', answer.answer, 'answer', answer.score);
        }
    }
    addGenericResult(key, value) {
        const displayValue = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
        this.addResultItem(key, displayValue, 'generic');
    }
    addResultItem(label, value, type = 'default', confidence = null) {
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
        this.elements.resultsContent.appendChild(item);
    }
    showLoading(message) {
        this.elements.resultsSection.style.display = 'block';
        this.elements.resultsContent.innerHTML = `
            <div class="result-item">
                <div class="result-label">
                    <span class="spinner"></span> ${message}
                </div>
            </div>
        `;
    }
    showError(message) {
        this.elements.resultsSection.style.display = 'block';
        this.elements.resultsContent.innerHTML = `
            <div class="result-item" style="background: #f8d7da; color: #721c24;">
                <div class="result-label">Error</div>
                <div class="result-value">${message}</div>
            </div>
        `;
    }
    hideResults() {
        this.elements.resultsSection.style.display = 'none';
    }
    showHelp() {
        this.elements.resultsSection.style.display = 'block';
        this.elements.resultsContent.innerHTML = `
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
}

// popup.ts - Popup interface bootstrap
// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.popupController = new PopupController();
});
// Listen for pipeline progress updates
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'pipeline-progress') {
        const controller = window.popupController;
        if (controller) {
            controller.checkPipelineStatus();
        }
    }
});
