/**
 * PageMonitor - Handles monitoring page changes and selection events
 */

import { UIManager } from './ui-manager';

export class PageMonitor {
    private extractionTimeout: number | undefined;
    private selectionHandler: ((event: Event) => void) | null = null;
    private onPageChange: () => void;

    constructor(onPageChange: () => void) {
        this.onPageChange = onPageChange;
    }

    /**
     * Setup page change monitoring
     */
    setupPageMonitoring(): void {
        const observer = new MutationObserver((mutations) => {
            let significantChange = false;

            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    for (const node of mutation.addedNodes) {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            const text = (node as Element).textContent || '';
                            if (text.length > 100) {
                                significantChange = true;
                                break;
                            }
                        }
                    }
                }
            });

            if (significantChange) {
                clearTimeout(this.extractionTimeout);
                this.extractionTimeout = window.setTimeout(() => {
                    this.onPageChange();
                }, 2000);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    /**
     * Setup text selection handling
     */
    setupSelectionHandling(analyzeCallback: (text: string) => Promise<any>): void {
        this.selectionHandler = (event: Event) => {
            const selection = window.getSelection();
            if (!selection || selection.toString().length === 0) return;

            const selectedText = selection.toString().trim();
            if (selectedText.length < 10) return;

            const mouseEvent = event as MouseEvent;
            UIManager.showSelectionTooltip(
                mouseEvent.clientX, 
                mouseEvent.clientY, 
                selectedText,
                analyzeCallback
            );
        };

        document.addEventListener('mouseup', this.selectionHandler);
    }

    /**
     * Cleanup event listeners
     */
    cleanup(): void {
        if (this.selectionHandler) {
            document.removeEventListener('mouseup', this.selectionHandler);
            this.selectionHandler = null;
        }

        if (this.extractionTimeout) {
            clearTimeout(this.extractionTimeout);
            this.extractionTimeout = undefined;
        }
    }
}
