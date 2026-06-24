export interface JSONSchemaObject {
    type: 'object';
    properties?: Record<string, JSONSchemaProperty>;
    required?: string[];
    additionalProperties?: boolean;
    description?: string;
}

export type JSONSchemaProperty =
    | { type: 'string'; description?: string; enum?: string[]; pattern?: string; default?: string }
    | { type: 'number'; description?: string; minimum?: number; maximum?: number; default?: number }
    | { type: 'integer'; description?: string; minimum?: number; maximum?: number; default?: number }
    | { type: 'boolean'; description?: string; default?: boolean }
    | { type: 'array'; description?: string; items?: JSONSchemaProperty }
    | { type: 'object'; description?: string; properties?: Record<string, JSONSchemaProperty>; required?: string[] };

export interface TextContent {
    type: 'text';
    text: string;
}

export interface ImageContent {
    type: 'image';
    data: string;
    mimeType: 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp';
}

export interface StructuredContent {
    type: 'structured';
    data: Record<string, unknown>;
}

export type ContentBlock = TextContent | ImageContent | StructuredContent;

export interface ToolCallOutput {
    content: ContentBlock[];
    isError?: boolean;
}

export interface ToolCall {
    name: string;
    args: Record<string, unknown>;
    id: string;
}

export interface ToolCallResult extends ToolCall {
    output: ToolCallOutput;
    durationMs: number;
}
