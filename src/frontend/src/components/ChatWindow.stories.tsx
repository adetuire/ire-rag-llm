import type { Meta, StoryObj } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import { ChatWindow } from './ChatWindow.tsx';

const meta: Meta<typeof ChatWindow> = {
  title: 'Chat/ChatWindow',
  component: ChatWindow,
  args: {
    messages: [
      {
        id: '1',
        role: 'user',
        content: 'What is the compliance summary for ACME Corp?',
        createdAt: new Date().toISOString()
      },
      {
        id: '2',
        role: 'assistant',
        content: 'ACME Corp is compliant with policies A and B. Policy C requires review.',
        createdAt: new Date().toISOString()
      }
    ],
    isLoading: false,
    onSendMessage: async (message: string) => action('sendMessage')(message)
  }
};

export default meta;

type Story = StoryObj<typeof ChatWindow>;

export const Default: Story = {};

export const Loading: Story = {
  args: {
    isLoading: true
  }
};

export const WithError: Story = {
  args: {
    error: 'Unable to reach the chat service.'
  }
};
