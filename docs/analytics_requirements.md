# Analytics & Observability Requirements

## Objectives
- Understand how users search, which documents drive successful answers, and where the assistant underperforms.
- Provide compliance teams with full lineage of responses for audits.
- Enable continuous improvement of retrieval and ranking via feedback loops.

## Data Capture Matrix
| Event | Trigger | Payload Fields | Destination |
| --- | --- | --- | --- |
| `query_issued` | User submits prompt | `session_id`, `message_id`, `user_id`, `prompt`, `filters`, `timestamp`, `client_app` | Event stream (Kafka topic `rag.events`) + OLAP warehouse |
| `response_stream_started` | Backend begins streaming answer | `session_id`, `message_id`, `rag_request_id`, `model_version`, `latency_ms`, `retriever_k` | Metrics service (Prometheus) |
| `document_retrieved` | Each document returned from retriever | `session_id`, `message_id`, `document_id`, `chunk_id`, `score`, `preview`, `metadata`, `rank` | Feature store + warehouse |
| `answer_completed` | Assistant finishes response | `session_id`, `message_id`, `tokens_prompt`, `tokens_completion`, `latency_ms`, `confidence` | Warehouse + monitoring |
| `clarification_shown` | Clarifying question surfaced | `session_id`, `message_id`, `clarification_id`, `options` | Warehouse |
| `clarification_selected` | User picks clarification | `session_id`, `message_id`, `clarification_id`, `selected_option` | Warehouse |
| `saved_answer_created` | User bookmarks response | `saved_answer_id`, `session_id`, `message_id`, `tags`, `notes_length` | Warehouse |
| `feedback_submitted` | Thumbs up/down or free-text feedback | `feedback_id`, `session_id`, `message_id`, `document_ids`, `rating`, `comment`, `submitted_by`, `timestamp` | Warehouse + alerting pipeline |
| `ingestion_job_completed` | Document pipeline finishes | `job_id`, `source_type`, `documents_processed`, `chunks_processed`, `errors`, `duration_ms` | Monitoring + warehouse |

## Storage & Processing
- **Event streaming**: Kafka (topic partitioned by `tenant_id`), 7-day retention; consumers write to warehouse (Snowflake/BigQuery) and feature store.
- **Metrics**: Prometheus counters/histograms for latency, throughput, error rates; Grafana dashboards for live monitoring.
- **Warehouse schema**: Star schema with fact tables (`fact_queries`, `fact_documents`, `fact_feedback`) and dimension tables (`dim_user`, `dim_document`, `dim_session`).
- **PII Handling**: Hash `user_id` and avoid storing raw prompts in lower environments; mask sensitive tokens via DLP.

## Relevance Feedback Loop
1. Store thumbs up/down plus granular feedback on individual documents.
2. Batch process nightly to compute document success rate, model confidence calibration, and filter usage patterns.
3. Feed aggregated metrics into re-ranking model training set and ingestion prioritization.
4. Provide dashboards for SMEs to review low-performing documents and submit updates.

## Audit & Compliance Reporting
- Generate weekly reports summarizing top regulatory queries, documents cited, and associated security tiers.
- Maintain immutable log of responses with metadata to satisfy GDPR/CCPA subject access requests.
- Implement retention policy: raw events 13 months, aggregated metrics 36 months, with region-specific overrides.

## Alerting & SLOs
- **SLO**: 99% of responses delivered < 5s; alert if rolling 5-min latency > threshold.
- Alert on ingestion failure rate > 2% per source in 1 hour.
- Alert when feedback negative ratio exceeds 15% for any product line within 24h.
- Trigger page if streaming errors > 5 per minute per region.

## Instrumentation Checklist
- Frontend logs interactions via Segment SDK (batched, sampling configurable).
- Backend middleware injects `session_id` and `rag_request_id` into structured logs.
- Python RAG service exports OpenTelemetry traces (`retrieve`, `generate`, `rerank` spans) with propagation from frontend request.
- All services share correlation IDs to stitch full journey across systems.
