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

export interface ToolCall {
    callID: string;
    name: string;
    arguments: Record<string, unknown>;
}

export type ToolResultContent =
    | { type: 'text'; value: string }
    | { type: 'image'; value: string | Blob | ArrayBuffer | Uint8Array }
    | { type: 'audio'; value: string | ArrayBuffer | Uint8Array }
    | { type: 'object'; value: unknown };

export interface ToolSuccess {
    callID: string;
    name: string;
    result: ToolResultContent[];
}

export interface ToolError {
    callID: string;
    name: string;
    errorMessage: string;
}

export type ToolResponse = ToolSuccess | ToolError;
