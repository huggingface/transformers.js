import type { ConstraintState } from './types';

type Expression =
    | { kind: 'empty' }
    | { kind: 'set'; bytes: Uint32Array }
    | { kind: 'choice'; choices: Expression[] }
    | { kind: 'sequence'; parts: Expression[] }
    | { kind: 'star'; child: Expression };

const EMPTY: Expression = { kind: 'empty' };
const encoder = new TextEncoder();

export function compileRegex(source: string): ConstraintState<number> {
    const machine = new RegexMachine(new RegexParser(source).parse());
    return {
        initial: machine.initial,
        transition: (state, byte) => machine.transition(state, byte),
        viable: (state) => state >= 0,
        accepting: (state) => machine.accepting(state),
        maskKey: (state) => machine.stateKey(state),
    };
}

class RegexParser {
    private index = 0;

    constructor(private readonly source: string) {}

    parse(): Expression {
        if (this.source.startsWith('^')) this.index++;
        const expression = this.alternation();
        if (this.peek() === '$' && this.index === this.source.length - 1) this.index++;
        if (this.index !== this.source.length) this.fail(`unexpected ${JSON.stringify(this.peek())}`);
        return expression;
    }

    private alternation(): Expression {
        const choices = [this.concatenation()];
        while (this.peek() === '|') {
            this.index++;
            choices.push(this.concatenation());
        }
        return choice(choices);
    }

    private concatenation(): Expression {
        const parts: Expression[] = [];
        while (this.index < this.source.length && this.peek() !== ')' && this.peek() !== '|') {
            if (this.peek() === '$' && this.index === this.source.length - 1) break;
            parts.push(this.quantified());
        }
        return sequence(parts);
    }

    private quantified(): Expression {
        const atom = this.atom();
        const quantifier = this.peek();
        if (!['*', '+', '?', '{'].includes(quantifier ?? '')) return atom;
        let minimum: number;
        let maximum: number | null;
        if (quantifier === '*') {
            this.index++;
            [minimum, maximum] = [0, null];
        } else if (quantifier === '+') {
            this.index++;
            [minimum, maximum] = [1, null];
        } else if (quantifier === '?') {
            this.index++;
            [minimum, maximum] = [0, 1];
        } else {
            [minimum, maximum] = this.bounds();
        }
        if (this.peek() === '?' || this.peek() === '+') this.fail('lazy and possessive quantifiers are unsupported');
        const parts = Array.from({ length: minimum }, () => atom);
        if (maximum === null) parts.push(star(atom));
        else for (let count = minimum; count < maximum; ++count) parts.push(choice([EMPTY, atom]));
        return sequence(parts);
    }

    private atom(): Expression {
        const character = this.peek();
        if (character === undefined) this.fail('expected an expression');
        if (character === '(') {
            this.index++;
            if (this.source.startsWith('?:', this.index)) this.index += 2;
            else if (this.peek() === '?') this.fail('only non-capturing special groups are supported');
            const result = this.alternation();
            if (this.peek() !== ')') this.fail('unterminated group');
            this.index++;
            return result;
        }
        if (character === '[') return this.characterClass();
        if (character === '.') {
            this.index++;
            return byteSet(range(0, 255));
        }
        if (character === '\\') return this.escape(false).expression;
        if ('*+?{})'.includes(character)) this.fail(`unexpected ${JSON.stringify(character)}`);
        this.index += character.length;
        return literal(character);
    }

    private characterClass(): Expression {
        this.index++;
        const negated = this.peek() === '^';
        if (negated) this.index++;
        const bytes = new Uint32Array(8);
        let hasValue = false;
        while (this.peek() !== ']') {
            if (this.peek() === undefined) this.fail('unterminated character class');
            const first = this.classValue();
            if (this.peek() === '-' && this.source[this.index + 1] !== ']') {
                this.index++;
                const last = this.classValue();
                if (first.single === undefined || last.single === undefined || first.single > last.single) {
                    this.fail('invalid character class range');
                }
                addRange(bytes, first.single, last.single);
            } else {
                union(bytes, first.bytes);
            }
            hasValue = true;
        }
        this.index++;
        if (!hasValue) this.fail('empty character class');
        if (negated) for (let word = 0; word < bytes.length; ++word) bytes[word] = ~bytes[word];
        return byteSet(bytes);
    }

    private classValue(): { bytes: Uint32Array; single?: number } {
        if (this.peek() === '\\') {
            const escaped = this.escape(true);
            if (escaped.bytes === undefined) this.fail('multi-byte escapes are unsupported in character classes');
            return { bytes: escaped.bytes, single: escaped.single };
        }
        const character = this.peek()!;
        this.index += character.length;
        const encoded = encoder.encode(character);
        if (encoded.length !== 1) this.fail('non-ASCII character classes are unsupported');
        return { bytes: singleton(encoded[0]), single: encoded[0] };
    }

    private escape(inClass: boolean): { expression: Expression; bytes?: Uint32Array; single?: number } {
        this.index++;
        const code = this.peek();
        if (code === undefined) this.fail('trailing escape');
        this.index++;
        if ('dDsSwW'.includes(code)) {
            const bytes = shorthand(code.toLowerCase());
            if (code === code.toUpperCase()) for (let word = 0; word < bytes.length; ++word) bytes[word] = ~bytes[word];
            return { expression: byteSet(bytes), bytes };
        }
        if (code === 'b' && !inClass) this.fail('word boundaries are unsupported');
        let value: number;
        if (code === 'x') value = this.hex(2);
        else if (code === 'u') value = this.hex(4);
        else
            value =
                ({ n: 10, r: 13, t: 9, f: 12, v: 11, b: 8 } as Record<string, number>)[code] ?? code.codePointAt(0)!;
        const text = String.fromCodePoint(value);
        const encoded = encoder.encode(text);
        const bytes = encoded.length === 1 ? singleton(encoded[0]) : undefined;
        return { expression: literal(text), bytes, single: encoded.length === 1 ? encoded[0] : undefined };
    }

    private bounds(): [number, number | null] {
        this.index++;
        const minimum = this.decimal();
        let maximum: number | null = minimum;
        if (this.peek() === ',') {
            this.index++;
            maximum = this.peek() === '}' ? null : this.decimal();
        }
        if (this.peek() !== '}') this.fail('unterminated repetition');
        this.index++;
        if (minimum > 1000 || (maximum !== null && (maximum < minimum || maximum > 1000))) {
            this.fail('invalid or excessive repetition');
        }
        return [minimum, maximum];
    }

    private decimal(): number {
        const start = this.index;
        while (/\d/.test(this.peek() ?? '')) this.index++;
        if (start === this.index) this.fail('expected a repetition count');
        return Number(this.source.slice(start, this.index));
    }

    private hex(length: number): number {
        const value = this.source.slice(this.index, this.index + length);
        if (!new RegExp(`^[\\da-f]{${length}}$`, 'i').test(value)) this.fail('invalid hexadecimal escape');
        this.index += length;
        return Number.parseInt(value, 16);
    }

    private peek(): string | undefined {
        return this.source[this.index];
    }

    private fail(message: string): never {
        throw new SyntaxError(`Invalid regex at index ${this.index}: ${message}.`);
    }
}

const OP_SET = 0;
const OP_SPLIT = 1;
const OP_JUMP = 2;
const OP_MATCH = 3;
const UNKNOWN = -2;

type Fragment = { start: number; outs: number[] };

class NfaBuilder {
    readonly ops: number[] = [];
    readonly out1: number[] = [];
    readonly out2: number[] = [];
    readonly sets: Uint32Array[] = [];

    compile(expression: Expression): Fragment {
        switch (expression.kind) {
            case 'empty': {
                const state = this.emit(OP_JUMP);
                return { start: state, outs: [state << 1] };
            }
            case 'set': {
                const state = this.emit(OP_SET, -1, -1, expression.bytes);
                return { start: state, outs: [state << 1] };
            }
            case 'sequence': {
                let result = this.compile(expression.parts[0]);
                for (let index = 1; index < expression.parts.length; ++index) {
                    const next = this.compile(expression.parts[index]);
                    this.patch(result.outs, next.start);
                    result = { start: result.start, outs: next.outs };
                }
                return result;
            }
            case 'choice': {
                let result = this.compile(expression.choices[0]);
                for (let index = 1; index < expression.choices.length; ++index) {
                    const right = this.compile(expression.choices[index]);
                    result = {
                        start: this.emit(OP_SPLIT, result.start, right.start),
                        outs: [...result.outs, ...right.outs],
                    };
                }
                return result;
            }
            case 'star': {
                const child = this.compile(expression.child);
                const split = this.emit(OP_SPLIT, child.start);
                this.patch(child.outs, split);
                return { start: split, outs: [(split << 1) | 1] };
            }
        }
    }

    emit(op: number, first = -1, second = -1, set?: Uint32Array): number {
        const state = this.ops.length;
        this.ops.push(op);
        this.out1.push(first);
        this.out2.push(second);
        this.sets.push(set ?? new Uint32Array(0));
        return state;
    }

    patch(outs: number[], target: number): void {
        for (const output of outs) {
            if (output & 1) this.out2[output >>> 1] = target;
            else this.out1[output >>> 1] = target;
        }
    }
}

class RegexMachine {
    readonly initial: number;
    private readonly builder = new NfaBuilder();
    private readonly states: number[][] = [];
    private readonly acceptingStates: boolean[] = [];
    private readonly transitionTables: Int32Array[] = [];
    private readonly stateIds = new Map<string, number>();
    private readonly stateKeys: string[] = [];

    constructor(expression: Expression) {
        const fragment = this.builder.compile(expression);
        const match = this.builder.emit(OP_MATCH);
        this.builder.patch(fragment.outs, match);
        this.initial = this.intern(this.closure([fragment.start]));
    }

    transition(state: number, byte: number): number {
        if (state < 0) return -1;
        const table = this.transitionTables[state];
        const cached = table[byte];
        if (cached !== UNKNOWN) return cached;
        const seeds: number[] = [];
        for (const pc of this.states[state]) {
            if (this.builder.ops[pc] !== OP_SET) continue;
            const set = this.builder.sets[pc];
            if (set[byte >>> 5] & (1 << (byte & 31))) seeds.push(this.builder.out1[pc]);
        }
        const next = seeds.length === 0 ? -1 : this.intern(this.closure(seeds));
        table[byte] = next;
        return next;
    }

    accepting(state: number): boolean {
        return state >= 0 && this.acceptingStates[state];
    }

    // The interned NFA state set is intrinsic to the regex (unlike the interned
    // ids, which depend on discovery order), so it is a stable mask-cache key
    // across constraint instances compiled from the same source.
    stateKey(state: number): string | undefined {
        return state >= 0 ? this.stateKeys[state] : undefined;
    }

    private closure(seeds: number[]): number[] {
        const result: number[] = [];
        const stack = [...seeds];
        const seen = new Set<number>();
        while (stack.length > 0) {
            const state = stack.pop()!;
            if (state < 0 || seen.has(state)) continue;
            seen.add(state);
            const op = this.builder.ops[state];
            if (op === OP_SPLIT) {
                stack.push(this.builder.out1[state], this.builder.out2[state]);
            } else if (op === OP_JUMP) {
                stack.push(this.builder.out1[state]);
            } else {
                result.push(state);
            }
        }
        result.sort((left, right) => left - right);
        return result;
    }

    private intern(active: number[]): number {
        const key = active.join(',');
        const existing = this.stateIds.get(key);
        if (existing !== undefined) return existing;
        if (this.states.length >= 4096) throw new Error('Regex produced too many runtime states.');
        const id = this.states.length;
        const transitions = new Int32Array(256);
        transitions.fill(UNKNOWN);
        this.states.push(active);
        this.acceptingStates.push(active.some((state) => this.builder.ops[state] === OP_MATCH));
        this.transitionTables.push(transitions);
        this.stateIds.set(key, id);
        this.stateKeys.push(key);
        return id;
    }
}

function choice(items: Expression[]): Expression {
    const flattened = items.flatMap((item) => (item.kind === 'choice' ? item.choices : [item]));
    if (flattened.length === 1) return flattened[0];
    return { kind: 'choice', choices: flattened };
}

function sequence(items: Expression[]): Expression {
    const flattened = items
        .flatMap((item) => (item.kind === 'sequence' ? item.parts : [item]))
        .filter((item) => item !== EMPTY);
    if (flattened.length === 0) return EMPTY;
    if (flattened.length === 1) return flattened[0];
    return { kind: 'sequence', parts: flattened };
}

function star(child: Expression): Expression {
    if (child === EMPTY) return EMPTY;
    if (child.kind === 'star') return child;
    return { kind: 'star', child };
}

function literal(value: string): Expression {
    return sequence([...encoder.encode(value)].map((byte) => byteSet(singleton(byte))));
}

function byteSet(bytes: Uint32Array): Expression {
    return { kind: 'set', bytes };
}

function shorthand(code: string): Uint32Array {
    const bytes = new Uint32Array(8);
    if (code === 'd' || code === 'w') addRange(bytes, 48, 57);
    if (code === 'w') {
        addRange(bytes, 65, 90);
        addRange(bytes, 97, 122);
        add(bytes, 95);
    }
    if (code === 's') for (const byte of [9, 10, 11, 12, 13, 32]) add(bytes, byte);
    return bytes;
}

function singleton(byte: number): Uint32Array {
    const bytes = new Uint32Array(8);
    add(bytes, byte);
    return bytes;
}

function range(first: number, last: number): Uint32Array {
    const bytes = new Uint32Array(8);
    addRange(bytes, first, last);
    return bytes;
}

function addRange(bytes: Uint32Array, first: number, last: number): void {
    for (let byte = first; byte <= last; ++byte) add(bytes, byte);
}

function add(bytes: Uint32Array, byte: number): void {
    bytes[byte >>> 5] |= 1 << (byte & 31);
}

function union(target: Uint32Array, source: Uint32Array): void {
    for (let word = 0; word < target.length; ++word) target[word] |= source[word];
}
