/**
 * Chrome Extension API Types
 */
export interface ChromeTab {
    id?: number;
    url?: string;
    title?: string;
    active?: boolean;
}
export interface ChromeMessageSender {
    tab?: ChromeTab;
    frameId?: number;
    id?: string;
    url?: string;
    origin?: string;
}
export type ChromeMessageHandler = (message: any, sender: ChromeMessageSender, sendResponse: (response: any) => void) => boolean | void;
export interface PageContent {
    title: string;
    mainContent: string;
    url: string;
    images: string[];
}
export interface PipelineStatus {
    loaded: string[];
    loading: string[];
    [taskType: string]: 'loading' | 'loaded' | 'error' | string[];
}
export interface ExtensionState {
    isInitialized: boolean;
    pipelines: Map<string, any>;
    status: Map<string, string>;
    cache: Map<string, any>;
}
//# sourceMappingURL=chrome.d.ts.map