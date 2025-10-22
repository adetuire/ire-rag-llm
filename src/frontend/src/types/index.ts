export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: string;
}

export interface DocumentPreview {
  id: string;
  title: string;
  snippet: string;
  sourceUrl?: string;
}

export interface SavedAnswer {
  id: string;
  question: string;
  answer: string;
  updatedAt: string;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
}
