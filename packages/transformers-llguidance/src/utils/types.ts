import type { LLGuidanceInterpreterHandle } from 'llguidance';

export type LlguidanceStats = {
    steps: number;
    computeMaskMs: number;
    applyMaskMs: number;
    commitTokenMs: number;
    trieNodesVisited: number;
    sharedJsonMaskCacheHits: number;
    sharedJsonMaskCacheMisses: number;
    stopReason?: 'accepted' | 'dead_end';
};

export type LlguidanceState = {
    completed: boolean;
    disposed: boolean;
    eosTokenIds: number[];
    interpreter: LLGuidanceInterpreterHandle;
    maskBuffer?: Uint32Array;
    step: number;
    stats: LlguidanceStats;
};
