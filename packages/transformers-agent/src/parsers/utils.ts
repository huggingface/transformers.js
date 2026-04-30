export function asRecord(value: unknown): Record<string, unknown> {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return {};
    }
    return value as Record<string, unknown>;
}

export function splitTopLevel(input: string, separator: string): string[] {
    const result: string[] = [];
    let start = 0;
    let depth = 0;

    for (let i = 0; i < input.length; i++) {
        const char = input[i];
        if (char === '{' || char === '[') depth += 1;
        if (char === '}' || char === ']') depth = Math.max(0, depth - 1);

        if (char === separator && depth === 0) {
            result.push(input.slice(start, i));
            start = i + 1;
        }
    }

    result.push(input.slice(start));
    return result.map((x) => x.trim()).filter(Boolean);
}
