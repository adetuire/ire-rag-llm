# React UI Flows & API Contracts

## Design Principles
- **Assistive first**: inline clarifications and source context reduce trial-and-error.
- **Progressive disclosure**: show concise previews by default, expand on demand.
- **Session aware**: persistent timeline for follow-up questions and saved answers.
- **Trustworthy**: every response ties to citations, metadata, and feedback controls.

## Conversational Search Flow
1. **Entry point** – User opens the "Ask" panel from CRM sidebar or standalone web app.
2. **Prompt composer** – Rich text box with suggested prompts, metadata filter chips (product, region, security tier) pulled from profile defaults.
3. **Response stream** – Model reply rendered with inline citations, collapse/expand previews, and "Save" / "Improve" actions.
4. **Follow-up controls** – Quick reply buttons ("Drill into policy", "Show implementation guide"), clarifying questions surfaced from backend suggestions.
5. **Conversation timeline** – Sticky left rail summarizing previous questions, enabling jump-back and sharing.

### Required Components
- `<ConversationLayout>` – orchestrates header (session info), main panel (messages), right rail (document viewer).
- `<MessageBubble>` – supports markdown rendering, inline code, citations, and status states (thinking, streaming, complete).
- `<FilterBar>` – multi-select chips bound to metadata facets; surfaces applied filters.
- `<DocumentPreviewDrawer>` – slides over to show full source with metadata.
- `<FeedbackBar>` – thumbs up/down, free-text comment, and "mark as authoritative" toggle for SMEs.

### API Contracts
| Endpoint | Method | Payload | Response |
| --- | --- | --- | --- |
| `/api/sessions/:sessionId/messages` | `POST` | `{ prompt: string, filters?: MetadataFilter, context?: MessageRef[] }` | `{ messageId, answer, citations: Citation[], documents: PreviewDocument[], clarifications?: ClarificationPrompt[] }` |
| `/api/sessions/:sessionId/messages` | `GET` | `?cursor=<timestamp>` | `{ messages: Message[], nextCursor? }` |
| `/api/metadata/facets` | `GET` | `?sessionId=...` | `{ facets: FacetDefinition[] }` |
| `/api/documents/:documentId` | `GET` | `?chunkId=` | `{ documentId, title, metadata, content, auditTrail }` |

`MetadataFilter` example:
```ts
{
  productLine?: string[];
  region?: ("global" | "amer" | "emea" | "apac")[];
  securityTier?: ("internal" | "confidential" | "restricted")[];
  lifecycleStage?: ("draft" | "active" | "deprecated")[];
}
```

`PreviewDocument` example:
```ts
{
  documentId: string;
  chunkId: string;
  title: string;
  preview: string;
  score: number;
  metadata: Record<string, string | string[]>;
  sourceUrl?: string;
}
```

## Clarification Flow
1. Backend detects ambiguous query (low confidence or multiple intents).
2. UI surfaces `<ClarificationCard>` under the assistant message with suggested follow-up questions.
3. User selects one suggestion or writes custom clarifying prompt; conversation timeline groups follow-up under parent.
4. On selection, new request includes `context` referencing original message IDs for disambiguation.

### Additional API Contract
| Endpoint | Method | Payload | Response |
| --- | --- | --- | --- |
| `/api/sessions/:sessionId/clarifications` | `POST` | `{ messageId, clarificationId }` | `{ messageId: string, answer: string, documents: PreviewDocument[] }` |

## Saved Answers Flow
1. User clicks "Save" on a message to bookmark response and associated documents.
2. Modal prompts for tags (`case`, `customer`, `topic`) and optional notes.
3. Saved answer appears in `<SavedAnswersPanel>` accessible from global nav, grouped by session or tag.
4. Users can export to PDF/email or share link with permissions enforced via backend.

### API Contracts
| Endpoint | Method | Payload | Response |
| --- | --- | --- | --- |
| `/api/sessions/:sessionId/saved-answers` | `POST` | `{ messageId, tags: string[], notes?: string }` | `{ savedAnswerId, savedAt }` |
| `/api/saved-answers` | `GET` | `?tag=&ownerId=` | `{ savedAnswers: SavedAnswerSummary[] }` |
| `/api/saved-answers/:savedAnswerId` | `GET` | - | `{ message, documents, metadata, notes }` |
| `/api/saved-answers/:savedAnswerId` | `DELETE` | - | `{ deleted: true }` |

## State Management & Data Layer
- Use **React Query** for request caching and optimistic updates on feedback or saved answers.
- Maintain conversation state in context provider (`ConversationProvider`) with reducer actions (`sendMessage`, `applyFilters`, `saveAnswer`).
- Stream responses via Server-Sent Events or WebSocket; update message state incrementally to enable streaming UI.

## Error & Empty States
- Show inline error banner with retry for network or backend failures.
- Provide "No documents found" state with quick actions (clear filters, broaden query).
- Escalate repeated failures by offering "Hand off to human" CTA linking to live support.
