import type { Meta, StoryObj } from '@storybook/react';
import { DocumentPreviewPane } from './DocumentPreviewPane.tsx';

const meta: Meta<typeof DocumentPreviewPane> = {
  title: 'Chat/DocumentPreviewPane',
  component: DocumentPreviewPane,
  args: {
    previews: [
      {
        id: '1',
        title: 'Policy A summary',
        snippet: 'Policy A outlines the controls required for data encryption...',
        sourceUrl: '#'
      },
      {
        id: '2',
        title: 'Policy B notes',
        snippet: 'Ensure annual review of vendor risk management documents.',
        sourceUrl: '#'
      }
    ]
  }
};

export default meta;

type Story = StoryObj<typeof DocumentPreviewPane>;

export const Default: Story = {};

export const Empty: Story = {
  args: { previews: [] }
};
