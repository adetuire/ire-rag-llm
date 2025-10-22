# Interface Specifications

## REST Endpoints (Node.js BFF & Django Orchestrator)

### Conversational APIs (Node.js BFF)
| Method | Path | Description | Auth | Payload | Response |
| --- | --- | --- | --- | --- | --- |
| POST | `/api/chat/messages` | Submit a user utterance and stream model response tokens. | Bearer (JWT) | `{ message: string, conversationId?: UUID, persona?: string }` | `200` stream of `{ type: "token" \| "complete", data: { messageId, content, citations[] } }` |
| GET | `/api/chat/conversations/:id` | Fetch conversation transcript, metadata, and retrieved documents. | Bearer (JWT) | n/a | `200` `{ conversation, messages[], retrievals[] }` |
| POST | `/api/chat/conversations` | Initialize a new conversation with optional seed context. | Bearer (JWT) | `{ title?: string, persona?: string, context?: string }` | `201` `{ conversationId }` |
| POST | `/api/chat/messages/:id/feedback` | Capture thumbs-up/down and free-form feedback. | Bearer (JWT) | `{ score: -1 \| 1, comment?: string }` | `202` `{ status: "queued" }` |
| POST | `/api/chat/messages/:id/actions/summarize` | Request post-hoc summarization. | Bearer (JWT) | `{ summaryType: "brief" \| "detailed" }` | `202` `{ jobId }` |

### Knowledge Management APIs (Django)
| Method | Path | Description | Auth | Payload | Response |
| --- | --- | --- | --- | --- | --- |
| POST | `/admin/documents` | Register a document ingestion job. | OAuth2 SSO | `{ title, sourceType, uri, tags[], complianceLevel }` | `202` `{ jobId, status }` |
| GET | `/admin/documents/:id` | Retrieve document metadata, ingestion status, and audit trail. | OAuth2 SSO | n/a | `200` `{ document, jobs[] }` |
| POST | `/admin/documents/:id/reingest` | Trigger re-ingestion for updated content. | OAuth2 SSO | `{ reason, priority }` | `202` `{ jobId }` |
| GET | `/admin/analytics/events` | Paginated analytics events with filters. | OAuth2 SSO | Query `{ eventType?, dateRange?, userId? }` | `200` `{ events[], nextCursor? }` |
| POST | `/admin/feature-flags` | Create or update feature flags shared with BFF/client. | OAuth2 SSO | `{ key, description, rollout, targetingRules[] }` | `200` `{ flag }` |

## GraphQL Schema (Node.js BFF)
```graphql
type Conversation {
  id: ID!
  title: String
  persona: String
  createdAt: DateTime!
  updatedAt: DateTime!
  messages(limit: Int = 50, before: ID): [Message!]!
  analytics: ConversationAnalytics
}

type Message {
  id: ID!
  role: MessageRole!
  content: String!
  citations: [Citation!]!
  latencyMs: Int
  retrievals: [RetrievalEvent!]!
  createdAt: DateTime!
}

type Citation {
  id: ID!
  document: Document!
  chunkId: ID!
  quote: String!
  sourceUrl: String
}

type RetrievalEvent {
  id: ID!
  score: Float!
  chunkId: ID!
  document: Document!
  rerankPosition: Int
}

type Document {
  id: ID!
  title: String!
  sourceType: String!
  tags: [String!]!
  complianceLevel: String
}

type ConversationAnalytics {
  totalTokens: Int!
  averageLatencyMs: Float!
  feedbackSummary: FeedbackAggregate!
}

type FeedbackAggregate {
  positive: Int!
  negative: Int!
  comments: [String!]!
}

type Query {
  conversation(id: ID!): Conversation
  conversations(cursor: ID, limit: Int = 20): [Conversation!]!
  searchDocuments(query: String!, limit: Int = 10): [Document!]!
}

type Mutation {
  sendMessage(input: SendMessageInput!): SendMessagePayload!
  provideFeedback(messageId: ID!, score: FeedbackScore!, comment: String): FeedbackPayload!
  startConversation(input: StartConversationInput): StartConversationPayload!
}

input SendMessageInput {
  conversationId: ID
  message: String!
  persona: String
  context: String
}

input StartConversationInput {
  title: String
  persona: String
  context: String
}

enum MessageRole { USER ASSISTANT SYSTEM }

enum FeedbackScore { POSITIVE NEGATIVE }

type SendMessagePayload {
  conversation: Conversation!
  message: Message!
}

type FeedbackPayload {
  message: Message!
  analytics: ConversationAnalytics
}

type StartConversationPayload {
  conversation: Conversation!
}

type Subscription {
  messageStream(conversationId: ID!): Message!
}
```

## React Component Hierarchy
```
AppShell
├─ AuthBoundary
│  └─ SessionProvider
├─ Layout
│  ├─ Sidebar
│  │  ├─ ConversationList
│  │  │  └─ ConversationListItem
│  │  └─ WorkspaceSwitcher
│  └─ MainPanel
│     ├─ ConversationHeader
│     ├─ ChatSurface
│     │  ├─ MessageList
│     │  │  ├─ MessageItem
│     │  │  │  ├─ MessageContent
│     │  │  │  └─ CitationBadges
│     │  │  └─ TypingIndicator
│     │  ├─ Composer
│     │  │  ├─ PromptTextarea
│     │  │  ├─ AttachmentPicker
│     │  │  └─ QuickActionButtons
│     │  └─ RetrievalInspector (collapsible)
│     ├─ AnalyticsPanel (lazy)
│     └─ FeedbackDrawer
└─ Modals
   ├─ DocumentDetailsModal
   ├─ FeatureFlagEditorModal
   └─ SettingsModal
```

## Event & Streaming Interfaces
- **WebSocket `wss://.../api/chat/stream`** – multiplexed channel delivering token streams and system notifications. Payload contract mirrors REST streaming response.
- **Server-Sent Events `/api/chat/events`** – fallback for legacy browsers; only emits conversation-level updates.
- **Kafka Topics** – `analytics.events.v1`, `ingestion.jobs.v1`, and `retrieval.feedback.v1` follow Avro schemas aligned with PostgreSQL table definitions.
