import { useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { useChatSession } from '../hooks/useChatSession.ts';
import ChatWindow from '../components/ChatWindow.tsx';
import DocumentPreviewPane from '../components/DocumentPreviewPane.tsx';
import { fetchDocumentPreview } from '../api/chat.ts';

export const ChatPage = () => {
  const { sessionId = 'default' } = useParams();
  const { messages, previews, isLoading, error, sendMessage } = useChatSession({ sessionId });

  const handleOpenSource = useCallback(async (preview: { id: string }) => {
    try {
      const fullPreview = await fetchDocumentPreview(preview.id);
      if (fullPreview.sourceUrl) {
        window.open(fullPreview.sourceUrl, '_blank', 'noopener');
      }
    } catch (err) {
      console.error('Failed to open document preview', err);
    }
  }, []);

  return (
    <div className="grid gap-4 lg:grid-cols-[2fr,1fr]">
      <div className="min-h-[540px]">
        <ChatWindow messages={messages} isLoading={isLoading} error={error} onSendMessage={sendMessage} />
      </div>
      <DocumentPreviewPane previews={previews} onOpenSource={handleOpenSource} />
    </div>
  );
};

export default ChatPage;
