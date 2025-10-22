import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ChatWindow } from '../src/components/ChatWindow.tsx';

const baseProps = {
  messages: [
    {
      id: '1',
      role: 'user' as const,
      content: 'Hello!',
      createdAt: new Date().toISOString()
    }
  ],
  onSendMessage: vi.fn()
};

describe('ChatWindow', () => {
  it('renders existing messages', () => {
    render(<ChatWindow {...baseProps} />);
    expect(screen.getByText('Hello!')).toBeInTheDocument();
  });

  it('submits new messages', async () => {
    const onSendMessage = vi.fn();
    render(<ChatWindow {...baseProps} onSendMessage={onSendMessage} />);

    const textarea = screen.getByPlaceholderText('Ask anything about your documents...');
    fireEvent.change(textarea, { target: { value: 'New question' } });
    fireEvent.submit(textarea.closest('form')!);

    expect(onSendMessage).toHaveBeenCalledWith('New question');
  });
});
