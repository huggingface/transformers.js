import type { JSONSchemaObject, JSONSchemaProperty, ToolResultContent } from './types';

type ParameterKey<TParams> = Extract<keyof TParams, string>;
type Awaitable<T> = T | Promise<T>;

export type ToolExecute<TParams extends Record<string, unknown>> = (args: TParams) => Awaitable<ToolResultContent[]>;

export type ToolParameter<TValue = unknown> = JSONSchemaProperty & {
    readonly __type?: TValue;
    readonly optional?: boolean;
};

export type ToolParameters<TParams extends Record<string, unknown>> = Partial<{
    [K in ParameterKey<TParams>]: ToolParameter<Exclude<TParams[K], undefined>>;
}>;

export interface ToolOptions<TParams extends Record<string, unknown>> {
    name: string;
    description: string;
    inputSchema?: JSONSchemaObject;
    parameters?: ToolParameters<TParams>;
    required?: ParameterKey<TParams>[];
    additionalProperties?: boolean;
    execute: ToolExecute<TParams>;
}

export interface ToolDeclaration {
    name: string;
    description: string;
    inputSchema: JSONSchemaObject;
}

export class Tool<TParams extends Record<string, unknown> = Record<string, unknown>> implements ToolDeclaration {
    readonly name: string;
    readonly description: string;
    readonly inputSchema: JSONSchemaObject;
    readonly execute: ToolExecute<TParams>;

    constructor(options: ToolOptions<TParams>) {
        this.name = options.name;
        this.description = options.description;
        this.inputSchema = options.inputSchema ?? Tool.createInputSchema(options);
        this.execute = options.execute;
    }

    static string(options: Omit<Extract<JSONSchemaProperty, { type: 'string' }>, 'type'> = {}): ToolParameter<string> {
        return { type: 'string', ...options };
    }

    static number(options: Omit<Extract<JSONSchemaProperty, { type: 'number' }>, 'type'> = {}): ToolParameter<number> {
        return { type: 'number', ...options };
    }

    static integer(
        options: Omit<Extract<JSONSchemaProperty, { type: 'integer' }>, 'type'> = {},
    ): ToolParameter<number> {
        return { type: 'integer', ...options };
    }

    static boolean(
        options: Omit<Extract<JSONSchemaProperty, { type: 'boolean' }>, 'type'> = {},
    ): ToolParameter<boolean> {
        return { type: 'boolean', ...options };
    }

    static array<TItem = unknown>(
        items?: ToolParameter<TItem>,
        options: Omit<Extract<JSONSchemaProperty, { type: 'array' }>, 'type' | 'items'> = {},
    ): ToolParameter<TItem[]> {
        return { type: 'array', ...options, ...(items ? { items: Tool.toJSONSchemaProperty(items) } : {}) };
    }

    static object<TObject extends Record<string, unknown> = Record<string, unknown>>(
        properties?: ToolParameters<TObject>,
        options: Omit<Extract<JSONSchemaProperty, { type: 'object' }>, 'type' | 'properties' | 'required'> & {
            required?: ParameterKey<TObject>[];
        } = {},
    ): ToolParameter<TObject> {
        const schemaProperties = Tool.toJSONSchemaProperties(properties ?? {});
        const required =
            options.required ??
            Object.entries(properties ?? {})
                .filter(([, parameter]) => parameter && !parameter.optional)
                .map(([name]) => name);
        return {
            type: 'object',
            ...options,
            ...(Object.keys(schemaProperties).length > 0 ? { properties: schemaProperties } : {}),
            ...(required.length > 0 ? { required } : {}),
        };
    }

    static optional<TValue>(parameter: ToolParameter<TValue>): ToolParameter<TValue | undefined> {
        return { ...parameter, optional: true };
    }

    private static createInputSchema<TParams extends Record<string, unknown>>(
        options: ToolOptions<TParams>,
    ): JSONSchemaObject {
        const properties = Tool.toJSONSchemaProperties(options.parameters ?? {});
        const required =
            options.required ??
            Object.entries(options.parameters ?? {})
                .filter(([, parameter]) => parameter && !parameter.optional)
                .map(([name]) => name);

        return {
            type: 'object',
            properties,
            required,
            additionalProperties: options.additionalProperties ?? false,
        };
    }

    private static toJSONSchemaProperties<TParams extends Record<string, unknown>>(
        parameters: ToolParameters<TParams>,
    ): Record<string, JSONSchemaProperty> {
        return Object.fromEntries(
            Object.entries(parameters).flatMap(([name, parameter]) => {
                if (!parameter) {
                    return [];
                }
                return [[name, Tool.toJSONSchemaProperty(parameter as ToolParameter)]];
            }),
        );
    }

    private static toJSONSchemaProperty(parameter: ToolParameter): JSONSchemaProperty {
        const { optional: _optional, __type: _type, ...schema } = parameter;
        return schema;
    }
}

export type ToolList = ToolDeclaration[];
