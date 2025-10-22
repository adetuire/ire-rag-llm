import { FormEvent, useState } from 'react';
import type { ChatMessage } from '../types/index.ts';

type ChatWindowProps = {
  messages: ChatMessage[];
  isLoading?: boolean;
  error?: string;
  onSendMessage: (message: string) => Promise<void> | void;
};

const roleStyles: Record<ChatMessage['role'], string> = {
  user: 'bg-primary-500 text-white self-end',
  assistant: 'bg-white border border-slate-200 text-slate-800 self-start',
  system: 'bg-slate-100 text-slate-600 self-center'
};

export const ChatWindow = ({ messages, isLoading, error, onSendMessage }: ChatWindowProps) => {
  const [value, setValue] = useState('');

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!value.trim()) return;
    await onSendMessage(value.trim());
    setValue('');
  };

  return (
    <div className="flex h-full flex-col gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex-1 overflow-y-auto rounded-lg bg-slate-50 p-4">
        <ul className="flex flex-col gap-3">
          {messages.map((message) => (
            <li
              key={message.id}
              className={`max-w-[80%] rounded-lg px-4 py-2 text-sm shadow-sm ${roleStyles[message.role]}`}
            >
              <div className="text-xs text-slate-500">{new Date(message.createdAt).toLocaleTimeString()}</div>
              <p className="whitespace-pre-line">{message.content}</p>
            </li>
          ))}
          {messages.length === 0 && (
            <li className="self-center rounded-lg bg-white px-3 py-2 text-sm text-slate-500">
              Start the conversation by asking a question.
            </li>
          )}
        </ul>
      </div>
      {error && <p className="text-sm text-red-500">{error}</p>}
      <form onSubmit={handleSubmit} className="flex items-end gap-3">
        <textarea
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Ask anything about your documents..."
          className="min-h-[80px] flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 focus:border-primary-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={isLoading}
          className="rounded-lg bg-primary-500 px-4 py-2 text-sm font-medium text-white shadow disabled:opacity-50"
        >
          {isLoading ? 'Sending…' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default ChatWindow;
