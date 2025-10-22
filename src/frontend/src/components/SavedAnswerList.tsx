import { useState } from 'react';
import type { SavedAnswer } from '../types/index.ts';

type SavedAnswerListProps = {
  answers: SavedAnswer[];
  onSelect?: (answer: SavedAnswer) => void;
  onDelete?: (answer: SavedAnswer) => Promise<void> | void;
  onUpdate?: (answer: SavedAnswer, updates: Partial<Pick<SavedAnswer, 'question' | 'answer'>>) => Promise<void> | void;
};

export const SavedAnswerList = ({ answers, onSelect, onDelete, onUpdate }: SavedAnswerListProps) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');

  return (
    <section className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">Saved answers</h2>
        <span className="text-xs text-slate-500">{answers.length} total</span>
      </div>
      <ul className="flex flex-col gap-3">
        {answers.map((answer) => (
          <li key={answer.id} className="rounded-lg border border-slate-200 p-3">
            <button
              type="button"
              className="text-sm font-semibold text-primary-600 hover:underline"
              onClick={() => onSelect?.(answer)}
            >
              {answer.question}
            </button>
            {editingId === answer.id ? (
              <div className="mt-2 flex flex-col gap-2">
                <textarea
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  className="min-h-[80px] w-full resize-none rounded border border-slate-300 px-3 py-2 text-sm"
                />
                <div className="flex gap-2">
                  <button
                    type="button"
                    className="rounded bg-primary-500 px-3 py-1 text-xs font-semibold text-white"
                    onClick={async () => {
                      if (editingId !== answer.id) return;
                      await onUpdate?.(answer, { answer: draft });
                      setEditingId(null);
                      setDraft('');
                    }}
                  >
                    Save
                  </button>
                  <button
                    type="button"
                    className="rounded border border-slate-300 px-3 py-1 text-xs font-semibold text-slate-600"
                    onClick={() => {
                      setEditingId(null);
                      setDraft('');
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <p className="mt-2 text-sm text-slate-600">{answer.answer}</p>
            )}
            <div className="mt-3 flex items-center gap-3 text-xs text-slate-500">
              <span>{new Date(answer.updatedAt).toLocaleString()}</span>
              <div className="ml-auto flex gap-2">
                <button
                  type="button"
                  className="text-primary-600 hover:underline"
                  onClick={() => {
                    setEditingId(answer.id);
                    setDraft(answer.answer);
                  }}
                >
                  Edit
                </button>
                <button
                  type="button"
                  className="text-red-500 hover:underline"
                  onClick={() => onDelete?.(answer)}
                >
                  Delete
                </button>
              </div>
            </div>
          </li>
        ))}
        {answers.length === 0 && (
          <li className="rounded-lg border border-dashed border-slate-300 p-4 text-sm text-slate-500">
            No saved answers yet.
          </li>
        )}
      </ul>
    </section>
  );
};

export default SavedAnswerList;
