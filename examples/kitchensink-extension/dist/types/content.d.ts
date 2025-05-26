/**
 * Page Content and Data Types
 */
export interface PageMetadata {
    title: string;
    url: string;
    domain: string;
    description: string;
    keywords: string[];
    language: string;
    wordCount: number;
}
export interface ImageData {
    src: string;
    alt: string;
    width: number;
    height: number;
    index: number;
}
export interface PageData {
    text: string;
    images: ImageData[];
    metadata: PageMetadata;
    lastAnalyzed: number | null;
}
export interface ContentAnalyzerConfig {
    isActive: boolean;
    selectionHandler: ((event: MouseEvent) => void) | null;
    pageData: PageData;
}
//# sourceMappingURL=content.d.ts.map