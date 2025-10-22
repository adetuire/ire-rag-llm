import { useCallback, useEffect } from 'react';
import {
  createSavedAnswer,
  deleteSavedAnswer,
  fetchSavedAnswers,
  fetchSessionMessages,
  sendMessage,
  updateSavedAnswer
} from '../api/chat.ts';
import type { SavedAnswer } from '../types/index.ts';
import { useChatStore } from '../store/chatStore.ts';

interface UseChatSessionOptions {
  sessionId?: string;
}

export const useChatSession = ({ sessionId }: UseChatSessionOptions = {}) => {
  const {
    messages,
    previews,
    savedAnswers,
    isLoading,
    error,
    setSessionId,
    setMessages,
    setPreviews,
    setSavedAnswers,
    setIsLoading,
    setError
  } = useChatStore();

  useEffect(() => {
    setIsLoading(true);
    setError(undefined);

    const loadSavedAnswers = fetchSavedAnswers()
      .then(setSavedAnswers)
      .catch((err: Error) => setError(err.message));

    if (!sessionId) {
      loadSavedAnswers.finally(() => setIsLoading(false));
      return;
    }

    setSessionId(sessionId);
    setPreviews([]);

    Promise.all([
      fetchSessionMessages(sessionId).then(setMessages),
      loadSavedAnswers
    ])
      .catch(() => {
        // error already captured in setError
      })
      .finally(() => setIsLoading(false));
  }, [sessionId, setSessionId, setIsLoading, setPreviews, setMessages, setSavedAnswers, setError]);

  const handleSendMessage = useCallback(
    async (content: string) => {
      if (!sessionId) {
        throw new Error('Session ID is required');
      }

      setIsLoading(true);
      setError(undefined);

      try {
        const response = await sendMessage({ sessionId, message: content });
        setMessages(response.messages);
        setPreviews(response.previews);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to send message';
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, setIsLoading, setMessages, setPreviews, setError]
  );

  const handleSaveAnswer = useCallback(
    async (answer: Pick<SavedAnswer, 'question' | 'answer'>) => {
      const created = await createSavedAnswer(answer);
      setSavedAnswers([...savedAnswers, created]);
      return created;
    },
    [savedAnswers, setSavedAnswers]
  );

  const handleUpdateAnswer = useCallback(
    async (id: string, answer: Partial<Pick<SavedAnswer, 'question' | 'answer'>>) => {
      const updated = await updateSavedAnswer(id, answer);
      setSavedAnswers(savedAnswers.map((item) => (item.id === id ? updated : item)));
      return updated;
    },
    [savedAnswers, setSavedAnswers]
  );

  const handleDeleteAnswer = useCallback(
    async (id: string) => {
      await deleteSavedAnswer(id);
      setSavedAnswers(savedAnswers.filter((item) => item.id !== id));
    },
    [savedAnswers, setSavedAnswers]
  );

  return {
    sessionId,
    messages,
    previews,
    savedAnswers,
    isLoading,
    error,
    sendMessage: handleSendMessage,
    saveAnswer: handleSaveAnswer,
    updateAnswer: handleUpdateAnswer,
    deleteAnswer: handleDeleteAnswer
  };
};
