import apiClient from './client.ts';
import type { ChatMessage, DocumentPreview, SavedAnswer } from '../types/index.ts';

export interface SendMessageRequest {
  sessionId: string;
  message: string;
}

export interface SendMessageResponse {
  messages: ChatMessage[];
  previews: DocumentPreview[];
}

export const sendMessage = async (
  payload: SendMessageRequest
): Promise<SendMessageResponse> => {
  const { data } = await apiClient.post<SendMessageResponse>('/chat/messages', payload);
  return data;
};

export const fetchSessionMessages = async (sessionId: string): Promise<ChatMessage[]> => {
  const { data } = await apiClient.get<ChatMessage[]>(`/chat/sessions/${sessionId}/messages`);
  return data;
};

export const fetchDocumentPreview = async (
  documentId: string
): Promise<DocumentPreview> => {
  const { data } = await apiClient.get<DocumentPreview>(`/documents/${documentId}`);
  return data;
};

export const fetchSavedAnswers = async (): Promise<SavedAnswer[]> => {
  const { data } = await apiClient.get<SavedAnswer[]>('/saved-answers');
  return data;
};

export const createSavedAnswer = async (
  payload: Pick<SavedAnswer, 'question' | 'answer'>
): Promise<SavedAnswer> => {
  const { data } = await apiClient.post<SavedAnswer>('/saved-answers', payload);
  return data;
};

export const updateSavedAnswer = async (
  id: string,
  payload: Partial<Pick<SavedAnswer, 'question' | 'answer'>>
): Promise<SavedAnswer> => {
  const { data } = await apiClient.patch<SavedAnswer>(`/saved-answers/${id}`, payload);
  return data;
};

export const deleteSavedAnswer = async (id: string): Promise<void> => {
  await apiClient.delete(`/saved-answers/${id}`);
};
