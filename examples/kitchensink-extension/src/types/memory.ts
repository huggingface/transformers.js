/**
 * Memory and Storage Types
 */

export interface MemoryEntry {
  type: string;
  content: string;
  timestamp: string;
  date: string;
}

export interface MemoryStats {
  pagesAnalyzed: number;
  insightsGenerated: number;
  entitiesFound: number;
  timeline: MemoryEntry[];
}
