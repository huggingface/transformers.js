/**
 * UIManager - Handles UI elements, tooltips, and analysis display
 */

import { AnalysisResult, Entity } from '../types';

export class UIManager {
    /**
     * Show selection analysis tooltip
     */
    static showSelectionTooltip(x: number, y: number, text: string, analyzeCallback: (text: string) => Promise<any>): void {
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
                const result = await analyzeCallback(text);
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

        // Auto-remove tooltip after 5 seconds
        setTimeout(() => {
            if (tooltip.parentNode) {
                tooltip.remove();
            }
        }, 5000);
    }

    /**
     * Inject analysis results UI panel
     */
    static injectAnalysisUI(data: any): void {
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

        // Auto-remove panel after 30 seconds
        setTimeout(() => {
            if (panel.parentNode) {
                panel.remove();
            }
        }, 30000);
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

    /**
     * Highlight named entities in the page content
     */
    static highlightNamedEntities(entities: Entity[]): void {
        if (!entities || entities.length === 0) return;

        const colors: Record<string, string> = {
            'PER': '#ffd6cc',    // Person
            'ORG': '#cce5ff',    // Organization  
            'LOC': '#ccffcc',    // Location
            'MISC': '#ffffcc'    // Miscellaneous
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
}
