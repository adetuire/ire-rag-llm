# Product Brief: Domain Knowledge Retrieval Platform

## Vision
Empower internal teams with a conversational assistant that answers questions with curated corporate knowledge while maintaining compliance and traceability across regional business units.

## Target Users
- **Customer support specialists** resolving policy and troubleshooting inquiries.
- **Account managers** preparing renewal proposals and executive summaries.
- **Solutions engineers** validating technical feasibility and implementation patterns.
- **Regional compliance leads** ensuring responses align with local regulations.

## Domain Document Catalogue

### Canonical Schema
| Field | Type | Description |
| --- | --- | --- |
| `document_id` | UUID | Stable identifier for the source document. |
| `source_type` | Enum (`playbook`, `policy`, `faq`, `runbook`, `case-study`) | High-level categorization for routing and compliance. |
| `title` | String | Human-readable title surfaced in search results. |
| `summary` | Text | Short synopsis used for previews. |
| `business_unit` | Enum | Team or organization owner (e.g., `support`, `sales`, `compliance`). |
| `product_line` | Enum | Product area or service that the document supports. |
| `region` | Enum (`global`, `amer`, `emea`, `apac`) | Regional applicability for filtering. |
| `language` | ISO-639-1 code | Primary language of the content. |
| `lifecycle_stage` | Enum (`draft`, `active`, `deprecated`) | Indicates freshness and retirement. |
| `effective_date` | Date | Start date for document validity. |
| `expires_at` | Date? | Optional end-of-life date. |
| `tags` | Array[String] | Folksonomy tags curated by SMEs. |
| `security_tier` | Enum (`internal`, `confidential`, `restricted`) | Access guardrails used by policy engine. |
| `author_id` | UUID | Owner for routing feedback. |
| `source_url` | URL | Link back to original system of record. |
| `version` | Integer | Incremented on each ingestion. |
| `checksum` | String | Hash used to detect content drift. |
| `blob_path` | String | Pointer to storage bucket object. |

### Chunk Schema (Embeddable Units)
| Field | Type | Description |
| --- | --- | --- |
| `chunk_id` | UUID | Unique chunk identifier. |
| `document_id` | UUID | Foreign key to canonical document. |
| `chunk_index` | Integer | Order of chunk within document. |
| `content` | Text | Cleaned text body used for retrieval. |
| `token_count` | Integer | Token length for embedding/backoff heuristics. |
| `embedding` | Vector[float] | Dense vector stored in FAISS or compatible index. |
| `metadata` | JSON | Flattened subset of document fields plus ingestion timestamp. |

## Ingestion Pipeline
1. **Source registration** – connectors for CMS exports (S3 drop), Confluence API, CRM attachments, and manual SME uploads.
2. **Fetch & normalize** – convert to markdown/plaintext, strip boilerplate, detect language, and classify document type.
3. **Metadata enrichment** – map ownership, region, lifecycle, security tier, and auto-tag using zero-shot classifier.
4. **Chunking** – recursive character text splitter (~750 tokens, 200 overlap) with adaptive chunking for tables and FAQs.
5. **Quality gates** – validation for required metadata, PII redaction, policy compliance, and deduplication via checksum.
6. **Embedding generation** – encode using `sentence-transformers/all-MiniLM-L6-v2`; store embeddings and metadata in vector store plus Postgres catalog.
7. **Publishing** – trigger vector index upsert, search cache warm-up, and Slack notifications for reviewers.
8. **Observability hooks** – emit ingestion metrics (`documents_processed`, `chunk_failures`, `embedding_latency`).

## Embedding Strategy
- **Primary model**: `sentence-transformers/all-MiniLM-L6-v2` for balanced latency/quality.
- **Specialized tuning**: fine-tune on labeled Q&A pairs for compliance-critical content; maintain experiments via MLflow.
- **Hybrid retrieval**: store BM25 indices (Elastic) alongside FAISS for lexical fallback on regulatory keywords.
- **Versioning**: maintain embedding `model_version` in metadata; re-embed delta documents asynchronously.
- **Security**: embeddings stored in dedicated namespace with row-level security keyed by `security_tier` and `region`.

## Must-Have User Journeys
1. **Answer customer question during live chat**
   - Trigger: Support specialist opens the assistant within the CRM sidebar.
   - Flow: Ask natural language question → filter by product line & region automatically → review top 3 answers with citations → insert curated response into chat.
   - Success metrics: reduced handle time, higher first-contact resolution.
2. **Prepare renewal brief**
   - Trigger: Account manager queries for latest ROI stories and compliance notes.
   - Flow: Search "renewal playbook" → clarify segment (enterprise vs SMB) → view saved answers and export summary deck.
   - Success metrics: time-to-insight, attach rate of cross-sell playbooks.
3. **Validate implementation feasibility**
   - Trigger: Solutions engineer evaluating custom feature request.
   - Flow: Ask for architecture precedent → apply metadata filters (`product_line`, `security_tier`) → inspect document previews → bookmark relevant designs.
   - Success metrics: reduced escalation volume, accuracy of feasibility assessments.
4. **Audit regulatory response**
   - Trigger: Compliance lead reviews assistant history for GDPR queries.
   - Flow: Filter conversations by region & lifecycle stage → inspect answer lineage with source metadata → provide feedback marking authoritative sources.
   - Success metrics: audit completeness, number of flagged inaccurate answers.

## Risks & Mitigations
- **Stale content** – enforce lifecycle metadata and automated re-ingestion alerts.
- **Access control gaps** – integrate document `security_tier` with IAM scopes and pre-filter retrieval.
- **Explainability** – include document previews, source URLs, and metadata in every answer for trust.
- **Adoption** – embed must-have journeys into CRM/Slack workflows with minimal context switching.
