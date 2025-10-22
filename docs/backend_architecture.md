# Backend Architecture & Routes

## Overview
The backend acts as an orchestration layer between client applications, the Python RAG service (`src/rag`), and upstream systems (document storage, auth, analytics). It should expose stable REST + streaming APIs, enforce security, and manage session context.

## Recommended Stack
- **Framework**: Node.js (NestJS) or Django REST Framework. Below outlines a Node/NestJS reference, with notes for Django parity.
- **Data stores**: Postgres for metadata/catalog, Redis for session cache & rate limiting, S3/GCS for document blobs.
- **Integration**: gRPC or REST proxy to Python RAG worker, message bus (Kafka) for ingestion triggers.

## Service Modules
1. **Auth & Identity** – integrates with SSO provider (OIDC). Injects user claims (role, region, business unit, security tier) into requests.
2. **Session Manager** – persists conversation sessions, message timeline, clarifications, and saved answers. Maintains TTL-based cache in Redis for active sessions.
3. **Document Orchestrator** – handles ingestion webhooks, manual uploads, and monitors pipeline jobs.
4. **RAG Proxy** – normalizes requests to Python RAG service, handles streaming responses, attaches metadata filters.
5. **Feedback & Analytics** – logs interactions, collects relevance feedback, exposes metrics endpoints.

## Route Blueprint (Node/NestJS style)
| Method | Path | Module | Description |
| --- | --- | --- | --- |
| `POST` | `/sessions` | Session Manager | Create new conversation session with inferred defaults (filters, persona). |
| `GET` | `/sessions/:sessionId` | Session Manager | Retrieve session metadata, participants, active filters. |
| `POST` | `/sessions/:sessionId/messages` | RAG Proxy | Forward user prompt and filters, stream assistant response + documents. |
| `GET` | `/sessions/:sessionId/messages` | Session Manager | Paginated history with citations & feedback state. |
| `POST` | `/sessions/:sessionId/clarifications` | RAG Proxy | Submit clarifying choice referencing parent message. |
| `POST` | `/sessions/:sessionId/saved-answers` | Session Manager | Bookmark a message and associated documents. |
| `GET` | `/saved-answers` | Session Manager | List saved answers with filters (owner, tag, session). |
| `DELETE` | `/saved-answers/:id` | Session Manager | Remove saved answer. |
| `POST` | `/documents/ingest` | Document Orchestrator | Trigger ingestion workflow (manual upload or connector sync). |
| `POST` | `/documents/ingest/webhook` | Document Orchestrator | Receive connector callbacks with payload metadata. |
| `GET` | `/documents/:documentId` | Document Orchestrator | Fetch document metadata, preview, audit trail. |
| `POST` | `/feedback` | Feedback & Analytics | Submit relevance feedback tied to message and document IDs. |
| `GET` | `/health` | Platform | Health/liveness probe exposing dependencies. |

### Streaming Implementation
- Expose `/sessions/:id/messages/stream` using Server-Sent Events (NestJS `@Sse` controller or Django Channels) to deliver incremental tokens.
- Attach metadata events for each retrieved document (`doc-preview`, `doc-update`).

### RAG Proxy Responsibilities
1. Translate REST payload into `RetrieveRequest` for Python service, injecting metadata filters derived from user claims.
2. Maintain circuit breaker & retries around Python worker; degrade gracefully to fallback search.
3. Normalize document previews (title, snippet, metadata) before returning to client.

## Document Ingestion Triggers
- **Manual upload**: `/documents/ingest` accepts multipart file, metadata JSON; enqueues job to ingestion pipeline (e.g., using BullMQ or Celery).
- **Scheduled sync**: Cron job hits external connectors, pushes delta payloads into queue.
- **Webhook**: `/documents/ingest/webhook` validates signature, maps payload to canonical schema, writes to staging table, and notifies pipeline service.

## Django Parity Notes
- Use Django REST Framework ViewSets with routers mirroring above endpoints.
- Streaming via Django Channels or HttpResponse `streaming_content` generator for SSE.
- Background jobs handled with Celery + Django-Q for ingestion tasks.
- Authentication using django-allauth or social-auth with JWT issuance; apply DRF permissions for `security_tier` filtering.

## Session Persistence Model
- `sessions` table: `session_id`, `user_id`, `created_at`, `active_filters (JSONB)`, `persona`.
- `messages` table: `message_id`, `session_id`, `role`, `content`, `citations (JSONB)`, `documents (JSONB)`, `created_at`, `model_latency_ms`.
- `clarifications` table: `clarification_id`, `message_id`, `prompt`, `selected_option`.
- `saved_answers` table: `id`, `session_id`, `message_id`, `tags`, `notes`, `created_by`, `created_at`.

## Security & Compliance Considerations
- Enforce attribute-based access control using `security_tier` & `region` claims on every request.
- Rate-limit prompts per user & per session using Redis token bucket.
- Audit logging for ingestion events and message access.
- Data retention policy configurable per region; soft-delete sessions older than N days while retaining analytics aggregates.
