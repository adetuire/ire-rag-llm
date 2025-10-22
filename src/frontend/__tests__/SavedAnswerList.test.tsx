import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { SavedAnswerList } from '../src/components/SavedAnswerList.tsx';

const answers = [
  {
    id: '1',
    question: 'Question?',
    answer: 'Answer',
    updatedAt: new Date().toISOString()
  }
];

describe('SavedAnswerList', () => {
  it('renders saved answers', () => {
    render(<SavedAnswerList answers={answers} />);
    expect(screen.getByText('Question?')).toBeInTheDocument();
  });

  it('invokes delete callback', () => {
    const onDelete = vi.fn();
    render(<SavedAnswerList answers={answers} onDelete={(answer) => onDelete(answer.id)} />);
    fireEvent.click(screen.getByText('Delete'));
    expect(onDelete).toHaveBeenCalledWith('1');
  });
});
