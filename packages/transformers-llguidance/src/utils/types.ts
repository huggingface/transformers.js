export type GuidanceMask = Uint32Array | Uint8Array | boolean[] | number[];

export type GuidanceMaskResult =
    | { mask: GuidanceMask; vocabSize?: number; stop?: false }
    | { stop: true }
    | { backtrack?: number; ffTokens?: number[] | Uint32Array };

export type GuidanceCommitResult = {
    stop?: boolean;
    backtrack?: number;
    ffTokens?: number[] | Uint32Array;
};

export type GuidanceMaskInstrumentation = {
    trieNodesVisited: number;
    sharedJsonMaskCacheHits: number;
    sharedJsonMaskCacheMisses: number;
};

export type GuidanceInterpreter = {
    computeMask(): GuidanceMaskResult;
    computeMaskInto?(target: Uint32Array): GuidanceMaskResult;
    maskInstrumentation?(): GuidanceMaskInstrumentation;
    commitToken(tokenId: number): GuidanceCommitResult | undefined;
};

export type LlguidanceStats = {
    steps: number;
    computeMaskMs: number;
    applyMaskMs: number;
    commitTokenMs: number;
    trieNodesVisited: number;
    sharedJsonMaskCacheHits: number;
    sharedJsonMaskCacheMisses: number;
};

export type LlguidanceState = {
    completed: boolean;
    interpreter: GuidanceInterpreter;
    maskBuffer?: Uint32Array;
    step: number;
    stats: LlguidanceStats;
};
