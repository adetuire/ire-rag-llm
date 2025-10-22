import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { ChatMessage, DocumentPreview, SavedAnswer } from '../types/index.ts';

interface ChatState {
  sessionId: string | null;
  messages: ChatMessage[];
  previews: DocumentPreview[];
  savedAnswers: SavedAnswer[];
  isLoading: boolean;
  error?: string;
  setSessionId: (sessionId: string) => void;
  setMessages: (messages: ChatMessage[]) => void;
  setPreviews: (previews: DocumentPreview[]) => void;
  setSavedAnswers: (answers: SavedAnswer[]) => void;
  setIsLoading: (value: boolean) => void;
  setError: (error?: string) => void;
}

export const useChatStore = create<ChatState>()(
  devtools((set) => ({
    sessionId: null,
    messages: [],
    previews: [],
    savedAnswers: [],
    isLoading: false,
    error: undefined,
    setSessionId: (sessionId) => set({ sessionId }),
    setMessages: (messages) => set({ messages }),
    setPreviews: (previews) => set({ previews }),
    setSavedAnswers: (savedAnswers) => set({ savedAnswers }),
    setIsLoading: (isLoading) => set({ isLoading }),
    setError: (error) => set({ error })
  }))
);
