/**
 * Content Script for AI Context Copilot
 * Initializes ContentAnalyzer for page interaction and analysis
 */

/// <reference types="chrome"/>

import { ContentAnalyzer } from './content-analyzer';

// Initialize content analyzer
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new ContentAnalyzer();
    });
} else {
    new ContentAnalyzer();
}
