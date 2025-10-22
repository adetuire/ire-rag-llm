import type { DocumentPreview } from '../types/index.ts';

type DocumentPreviewPaneProps = {
  previews: DocumentPreview[];
  onOpenSource?: (preview: DocumentPreview) => void;
};

export const DocumentPreviewPane = ({ previews, onOpenSource }: DocumentPreviewPaneProps) => (
  <aside className="flex h-full w-full flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
    <h2 className="text-sm font-semibold text-slate-700">Document previews</h2>
    {previews.length === 0 ? (
      <p className="text-sm text-slate-500">Relevant sources will appear here as you chat.</p>
    ) : (
      <ul className="flex flex-1 flex-col gap-3 overflow-y-auto">
        {previews.map((preview) => (
          <li key={preview.id} className="rounded-lg border border-slate-200 p-3">
            <h3 className="text-sm font-semibold text-slate-700">{preview.title}</h3>
            <p className="mt-1 text-xs text-slate-600">{preview.snippet}</p>
            {preview.sourceUrl && (
              <button
                type="button"
                className="mt-3 text-xs font-medium text-primary-600 hover:underline"
                onClick={() => onOpenSource?.(preview)}
              >
                View source
              </button>
            )}
          </li>
        ))}
      </ul>
    )}
  </aside>
);

export default DocumentPreviewPane;
