export interface ProgressData {
    progress: number;
    loaded: number;
    total: number;
}

export type ProgressCallback = (data: ProgressData) => void;
