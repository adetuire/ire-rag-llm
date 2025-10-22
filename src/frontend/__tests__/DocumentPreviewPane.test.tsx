import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { DocumentPreviewPane } from '../src/components/DocumentPreviewPane.tsx';

const previews = [
  {
    id: '1',
    title: 'Doc 1',
    snippet: 'Snippet 1',
    sourceUrl: '#'
  }
];

describe('DocumentPreviewPane', () => {
  it('shows empty state', () => {
    render(<DocumentPreviewPane previews={[]} />);
    expect(screen.getByText('Relevant sources will appear here as you chat.')).toBeInTheDocument();
  });

  it('lists document previews', () => {
    render(<DocumentPreviewPane previews={previews} onOpenSource={vi.fn()} />);
    expect(screen.getByText('Doc 1')).toBeInTheDocument();
  });
});
