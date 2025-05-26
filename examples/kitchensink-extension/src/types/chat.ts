/**
 * Chat and Conversation Types
 */

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
}
