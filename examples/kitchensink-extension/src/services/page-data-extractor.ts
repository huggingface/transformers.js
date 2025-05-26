/**
 * PageDataExtractor - Handles extracting content and metadata from web pages
 */

import { PageMetadata, ImageData } from '../types';

export class PageDataExtractor {
    /**
     * Extract main text content from the page
     */
    static extractMainText(): string {
        const mainSelectors = [
            'main', 'article', '[role="main"]', '.main-content', 
            '#main-content', '.content', '#content', '.post-content',
            '.article-content', '.entry-content'
        ];

        const removeSelectors = [
            'nav', 'header', 'footer', 'aside', '.sidebar', 
            '.navigation', '.menu', 'script', 'style', 
            '.advertisement', '.ads', '.social-share'
        ];

        // Try to find main content area
        let mainElement: Element | null = null;
        for (const selector of mainSelectors) {
            const element = document.querySelector(selector);
            if (element) {
                mainElement = element;
                break;
            }
        }

        // Fallback to body if no main content found
        if (!mainElement) {
            mainElement = document.body;
        }

        let text = mainElement.innerText || mainElement.textContent || '';

        // Remove unwanted content
        removeSelectors.forEach(selector => {
            const elements = mainElement!.querySelectorAll(selector);
            elements.forEach(el => {
                const elementText = (el as HTMLElement).innerText || '';
                text = text.replace(elementText, '');
            });
        });

        // Clean up whitespace and limit length
        text = text.replace(/\s+/g, ' ').trim();
        return text.substring(0, 10000);
    }

    /**
     * Extract images from the page
     */
    static extractImages(): ImageData[] {
        const images: ImageData[] = [];
        const imgElements = document.querySelectorAll('img');
        
        imgElements.forEach((img, index) => {
            if (img.src && img.width > 50 && img.height > 50) {
                images.push({
                    src: img.src,
                    alt: img.alt || '',
                    width: img.width,
                    height: img.height,
                    index: index
                });
            }
        });

        return images.slice(0, 10); // Limit to 10 images
    }

    /**
     * Extract page metadata
     */
    static extractMetadata(textContent: string): PageMetadata {
        const metadata: PageMetadata = {
            title: document.title,
            url: window.location.href,
            domain: window.location.hostname,
            description: '',
            keywords: [],
            language: document.documentElement.lang || 'en',
            wordCount: textContent ? textContent.split(' ').length : 0
        };

        // Extract description meta tag
        const descMeta = document.querySelector('meta[name="description"]') as HTMLMetaElement;
        if (descMeta) {
            metadata.description = descMeta.getAttribute('content') || '';
        }

        // Extract keywords meta tag
        const keywordsMeta = document.querySelector('meta[name="keywords"]') as HTMLMetaElement;
        if (keywordsMeta) {
            metadata.keywords = keywordsMeta.getAttribute('content')?.split(',').map(k => k.trim()) || [];
        }

        return metadata;
    }

    /**
     * Get currently selected text
     */
    static getSelectedText(): string {
        const selection = window.getSelection();
        return selection ? selection.toString().trim() : '';
    }
}
