export type JSONValue = null | boolean | number | string | JSONValue[] | { [key: string]: JSONValue };
export type JSONSchema = boolean | { [key: string]: JSONValue };
export type TokenizerSource = object | ((...args: unknown[]) => unknown);

export interface ConstraintState<State> {
    readonly initial: State;
    initialize?(bytes: Uint8Array): State;
    transition(state: State, byte: number): State;
    viable(state: State): boolean;
    accepting(state: State): boolean;
    /** Whether generation may end here with an end token whose own bytes are `bytes`. Defaults to `accepting`. */
    acceptsEnd?(state: State, bytes: Uint8Array): boolean;
    allowsSpecial?(state: State, next: State, bytes: Uint8Array): boolean;
    stringCapacity?(state: State): number | undefined;
    maskKey?(state: State): string | undefined;
}
