// popup.ts - Popup interface bootstrap
/// <reference types="chrome"/>

import { PopupController } from './popup-controller';

// Make controller available globally for message listener
declare global {
    interface Window {
        popupController?: PopupController;
    }
}

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
