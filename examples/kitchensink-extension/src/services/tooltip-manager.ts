/**
 * TooltipManager - Handles selection tooltips and analysis display
 */

import { AnalysisResult } from '../types';

export class TooltipManager {
    /**
     * Show selection analysis tooltip
     */
    static showSelectionTooltip(x: number, y: number, text: string, analyzeCallback: (text: string) => Promise<any>): void {
        const existingTooltip = document.getElementById('ai-copilot-tooltip');
        if (existingTooltip) {
            existingTooltip.remove();
        }

        if (text.length < 10) return;

        const tooltip = this.createTooltipElement(x, y);
        document.body.appendChild(tooltip);
        this.attachTooltipHandler(tooltip, text, analyzeCallback);
        this.scheduleTooltipRemoval(tooltip);
    }

    private static createTooltipElement(x: number, y: number): HTMLDivElement {
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
        return tooltip;
    }

    private static attachTooltipHandler(tooltip: HTMLDivElement, text: string, analyzeCallback: (text: string) => Promise<any>): void {
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
    }

    private static scheduleTooltipRemoval(tooltip: HTMLDivElement): void {
        setTimeout(() => {
            if (tooltip.parentNode) {
                tooltip.remove();
            }
        }, 5000);
    }
}
