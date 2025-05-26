/**
 * AnalysisPanelManager - Handles analysis result panels and data formatting
 */

import { AnalysisResult } from '../types';

export class AnalysisPanelManager {
    /**
     * Inject analysis results UI panel
     */
    static injectAnalysisUI(data: any): void {
        const existingPanel = document.getElementById('ai-copilot-panel');
        if (existingPanel) {
            existingPanel.remove();
        }

        const panel = this.createAnalysisPanel(data);
        document.body.appendChild(panel);
        this.schedulePanelRemoval(panel);
    }

    private static createAnalysisPanel(data: any): HTMLDivElement {
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
        return panel;
    }

    /**
     * Format analysis data for display
     */
    private static formatAnalysisData(data: any): string {
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

    private static schedulePanelRemoval(panel: HTMLDivElement): void {
        setTimeout(() => {
            if (panel.parentNode) {
                panel.remove();
            }
        }, 30000);
    }
}
