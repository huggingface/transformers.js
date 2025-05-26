/**
 * EntityHighlighter - Handles highlighting of named entities in page content
 */

import { NamedEntity } from '../types';

export class EntityHighlighter {
    private static readonly colors: Record<string, string> = {
        'PER': '#ffd6cc',    // Person
        'ORG': '#cce5ff',    // Organization  
        'LOC': '#ccffcc',    // Location
        'MISC': '#ffffcc'    // Miscellaneous
    };

    /**
     * Highlight named entities in the page content
     */
    static highlightNamedEntities(entities: NamedEntity[]): void {
        if (!entities || entities.length === 0) return;

        const textNodes = this.getTextNodes();
        entities.forEach(entity => {
            this.highlightEntity(entity, textNodes);
        });
    }

    private static getTextNodes(): Text[] {
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT
        );

        const textNodes: Text[] = [];
        let node: Node | null;
        while (node = walker.nextNode()) {
            textNodes.push(node as Text);
        }

        return textNodes;
    }

    private static highlightEntity(entity: NamedEntity, textNodes: Text[]): void {
        const entityText = entity.word || entity.entity_group;
        const color = this.colors[entity.entity_group] || this.colors['MISC'];

        textNodes.forEach(textNode => {
            const text = textNode.textContent || '';
            const index = text.indexOf(entityText);

            if (index !== -1) {
                this.replaceTextWithHighlight(textNode, entityText, color, entity, index);
            }
        });
    }

    private static replaceTextWithHighlight(
        textNode: Text, 
        entityText: string, 
        color: string, 
        entity: NamedEntity, 
        index: number
    ): void {
        const parent = textNode.parentNode;
        if (!parent) return;

        const text = textNode.textContent || '';
        const before = text.substring(0, index);
        const after = text.substring(index + entityText.length);

        const entitySpan = this.createEntitySpan(entityText, color, entity);
        const beforeNode = document.createTextNode(before);
        const afterNode = document.createTextNode(after);

        parent.insertBefore(beforeNode, textNode);
        parent.insertBefore(entitySpan, textNode);
        parent.insertBefore(afterNode, textNode);
        parent.removeChild(textNode);
    }

    private static createEntitySpan(entityText: string, color: string, entity: NamedEntity): HTMLSpanElement {
        const entitySpan = document.createElement('span');
        entitySpan.textContent = entityText;
        entitySpan.style.backgroundColor = color;
        entitySpan.style.padding = '1px 2px';
        entitySpan.style.borderRadius = '2px';
        entitySpan.title = `${entity.entity_group}: ${Math.round(entity.score * 100)}% confidence`;
        return entitySpan;
    }
}
