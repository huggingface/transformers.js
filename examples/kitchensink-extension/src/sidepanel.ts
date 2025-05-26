/**
 * Side Panel bootstrap for AI Context Copilot
 * Initializes SidePanelController for the main interface
 */

/// <reference types="chrome"/>

import { SidePanelController } from './sidepanel-controller';

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new SidePanelController();
    });
} else {
    new SidePanelController();
}
