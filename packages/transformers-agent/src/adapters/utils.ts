export function asRecord(value: unknown): Record<string, unknown> {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return {};
    }
    return value as Record<string, unknown>;
}

export function splitTopLevel(input: string, delimiter: string): string[] {
    const parts: string[] = [];
    let current = '';
    let depth = 0;
    let quote: string | null = null;

    for (let i = 0; i < input.length; i++) {
        const char = input[i];
        if (quote) {
            current += char;
            if (char === quote && input[i - 1] !== '\\') {
                quote = null;
            }
            continue;
        }
        if (char === '"' || char === "'") {
            quote = char;
            current += char;
            continue;
        }
        if (char === '{' || char === '[' || char === '(') depth += 1;
        if (char === '}' || char === ']' || char === ')') depth -= 1;
        if (char === delimiter && depth === 0) {
            parts.push(current.trim());
            current = '';
            continue;
        }
        current += char;
    }

    if (current.trim()) {
        parts.push(current.trim());
    }
    return parts;
}
