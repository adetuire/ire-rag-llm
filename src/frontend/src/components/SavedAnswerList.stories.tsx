import type { Meta, StoryObj } from '@storybook/react';
import { SavedAnswerList } from './SavedAnswerList.tsx';

const meta: Meta<typeof SavedAnswerList> = {
  title: 'Chat/SavedAnswerList',
  component: SavedAnswerList,
  args: {
    answers: [
      {
        id: '1',
        question: 'How do we comply with SOC 2?',
        answer: 'Follow the SOC 2 controls documented in the compliance playbook.',
        updatedAt: new Date().toISOString()
      },
      {
        id: '2',
        question: 'Where is the vendor security checklist?',
        answer: 'It is stored in the shared drive under /security/vendor-checklist.',
        updatedAt: new Date().toISOString()
      }
    ]
  }
};

export default meta;

type Story = StoryObj<typeof SavedAnswerList>;

export const Default: Story = {};

export const Empty: Story = {
  args: { answers: [] }
};
