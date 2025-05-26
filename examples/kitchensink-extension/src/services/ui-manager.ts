/**
 * UIManager - Main coordinator for UI components
 */

import { NamedEntity } from '../types';
import { TooltipManager } from './tooltip-manager';
import { AnalysisPanelManager } from './analysis-panel-manager';
import { EntityHighlighter } from './entity-highlighter';

export class UIManager {
    /**
     * Show selection analysis tooltip
     */
    static showSelectionTooltip(x: number, y: number, text: string, analyzeCallback: (text: string) => Promise<any>): void {
        TooltipManager.showSelectionTooltip(x, y, text, analyzeCallback);
    }

    /**
     * Inject analysis results UI panel
     */
    static injectAnalysisUI(data: any): void {
        AnalysisPanelManager.injectAnalysisUI(data);
    }

    /**
     * Highlight named entities in the page content
     */
    static highlightNamedEntities(entities: NamedEntity[]): void {
        EntityHighlighter.highlightNamedEntities(entities);
    }
}
