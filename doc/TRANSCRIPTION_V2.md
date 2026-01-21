# Transcription V2 Service

## Overview

Transcription V2 enhances V1 transcription output through normalization and LLM-based classification, producing structured segment insights.

**Pipeline:**
1. **Normalization** (Algorithmic) - Remove duplicates, resolve overlaps, merge fragments
2. **Classification** (LLM) - Identify topics and segment types
3. **Publish** - Send to RabbitMQ for downstream consumers

**Service is stateless** - does not store in database, only publishes to queue.

---

## LLM Client with Fallback

The enhanced LLM client supports **mixed-format fallback chains** with automatic provider detection.

### Supported Formats

#### 1. Model Names (Auto-Detection)
```python
provider_chain = ["gemini-2.5-flash", "gpt-4o", "gemini-3.0-pro"]
```
- `gemini-*` → Gemini provider
- `gpt-*`, `o1-*` → OpenAI provider
- `claude-*` → Anthropic provider (future)

#### 2. Provider Names (Uses Defaults)
```python
provider_chain = ["gemini", "openai", "azure"]
```
- Uses default models from config

#### 3. Explicit Dicts (Custom Models)
```python
provider_chain = [
    {"provider": "azure", "model": "my-custom-gpt4-deployment"},
    {"provider": "gemini", "model": "gemini-2.0-flash-thinking-exp"},
]
```

#### 4. Mixed Format (Recommended)
```python
provider_chain = [
    "gemini-2.5-flash",                                # Auto-detect
    {"provider": "openai", "model": "gpt-4o-mini"},    # Explicit
    "azure"                                             # Provider-level
]
```

### Configuration

**Environment Variables:**
```bash
# Default fallback chain (comma-separated)
LLM_FALLBACK_CHAIN="gemini-2.5-flash,gpt-4o,azure"

# LLM settings
LLM_TIMEOUT=1200.0
LLM_MAX_RETRIES=3

# Provider credentials
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

### Usage Examples

#### Example 1: Use Config Default
```python
from app.core.llm_client import llm_client

# Uses LLM_FALLBACK_CHAIN from config
response = await llm_client.chat_completion(
    messages=[{"role": "user", "content": "Summarize this meeting"}],
    temperature=0.3,
    max_tokens=8192
)
```

#### Example 2: Override with Model Names
```python
# Try specific models in sequence
response = await llm_client.chat_completion(
    messages=[{"role": "user", "content": "Classify these segments"}],
    provider_chain=["gemini-2.5-flash", "gpt-4o", "gpt-4o-mini"],
    response_format={"type": "json_object"}
)
```

#### Example 3: Mixed Format
```python
# Combine auto-detection and custom configs
response = await llm_client.chat_completion(
    messages=[...],
    provider_chain=[
        "gemini-2.5-flash",                            # Try Gemini 2.5 flash first
        {"provider": "openai", "model": "gpt-4o"},     # Then OpenAI GPT-4o
        {"provider": "azure", "model": "my-deployment"} # Finally Azure custom
    ]
)
```

#### Example 4: Global Model Override
```python
# Override all chain models with specific model
response = await llm_client.chat_completion(
    messages=[...],
    provider_chain=["gemini", "openai"],  # Provider-level chain
    model="gpt-4o-mini"                   # Force all to use this model
)
# Result: tries gemini with gpt-4o-mini, then openai with gpt-4o-mini
```

### Fallback Behavior

**Example: What happens when gemini-2.5-flash fails**

Config: `LLM_FALLBACK_CHAIN="gemini-2.5-flash,gpt-4o,gemini-3.0-pro"`

1. **Try Gemini with gemini-2.5-flash**
   - Retry 3 times with exponential backoff (1s, 2s, 4s)
   - If all retries fail → next in chain

2. **Try OpenAI with gpt-4o**
   - Retry 3 times with exponential backoff
   - If all retries fail → next in chain

3. **Try Gemini with gemini-3.0-pro**
   - Retry 3 times with exponential backoff
   - If all retries fail → raise Exception

**Total attempts:** 9 (3 models × 3 retries each)

---

## Transcription V2 Usage

### Basic Usage
```python
from app.services.transcription_v2_service import transcription_v2_service

# V1 transcription from existing service
v1_transcription = {
    "transcriptions": [
        {
            "segment_id": "seg_1",
            "start": 0.0,
            "end": 5.2,
            "text": "Let's discuss the project timeline.",
            "speaker": "John",
            "sentiment": "neutral"
        },
        # ... more segments
    ]
}

# Process and publish V2
result = await transcription_v2_service.process_and_publish(
    v1_transcription=v1_transcription,
    meeting_id="meeting_123",
    tenant_id="tenant_456",
    platform="offline"
)

print(f"Status: {result['status']}")
print(f"Message ID: {result['message_id']}")
```

### With Custom Options
```python
result = await transcription_v2_service.process_and_publish(
    v1_transcription=v1_transcription,
    meeting_id="meeting_123",
    tenant_id="tenant_456",
    platform="offline",
    options={
        "normalization": {
            "duplicate_similarity_threshold": 0.85,
            "min_fragment_duration": 2.0,
            "max_merge_gap": 1.0,
            "remove_filler_words": False
        },
        "classification": {
            "provider_chain": ["gemini-2.5-flash", "gpt-4o"],
            "temperature": 0.3,
            "max_tokens": 8192
        }
    }
)
```

### RabbitMQ Output

Published to queue: `transcription.v2.ready`

```json
{
  "metadata": {
    "messageId": "uuid",
    "timestamp": "2026-01-14T10:30:00Z"
  },
  "payload": {
    "meeting_id": "meeting_123",
    "tenant_id": "tenant_456",
    "platform": "offline",
    "status": "completed",
    "segment_classifications": {
      "clusters": [
        {
          "topic": "Project timeline discussion",
          "type": "decision",
          "segment_ids": ["seg_1", "seg_2"],
          "segments": [...],
          "metadata": {
            "start_time": 0.0,
            "end_time": 10.5,
            "duration": 10.5,
            "segment_count": 2,
            "speakers": ["John", "Sarah"]
          }
        }
      ],
      "metadata": {
        "total_clusters": 5,
        "clusters_by_type": {
          "decision": 2,
          "actionable_item": 1,
          "key_insight": 1,
          "question": 1
        }
      }
    },
    "processing_stats": {
      "normalization": {
        "original_segment_count": 25,
        "normalized_segment_count": 20,
        "segments_removed": 5
      },
      "classification": {
        "total_clusters": 5,
        "clusters_by_type": {...}
      }
    }
  }
}
```

### Segment Classification Types

- **actionable_item**: Tasks, action items, assignments, TODOs
- **decision**: Decisions made, conclusions reached, agreements
- **key_insight**: Important insights, discoveries, realizations
- **question**: Questions asked (answered or unanswered)

*(general_discussion is classified but filtered out from output)*

---

## File Structure

```
app/
├── core/
│   ├── config.py                    # Configuration with LLM settings
│   ├── llm_client.py                # Enhanced LLM client with fallback
│   └── openai_client.py             # Legacy client (deprecated for V2)
│
├── services/
│   ├── transcription_v2_service.py  # Main V2 orchestrator
│   └── processors/
│       ├── normalization_processor.py    # Algorithmic normalization
│       ├── classification_processor.py   # LLM-based classification
│       └── cluster_builder.py            # Cluster structure builder
│
└── messaging/
    └── producers/
        └── transcription_v2_producer.py  # RabbitMQ publisher
```

---

## Development Notes

### Adding New LLM Providers

1. Add client initialization in `LLMClient._initialize_clients()`
2. Add provider detection in `LLMClient._normalize_chain_item()`
3. Add client/model mapping in `LLMClient._get_provider_client_and_model()`

### Testing Fallback

```python
# Test fallback by using invalid first model
response = await llm_client.chat_completion(
    messages=[...],
    provider_chain=[
        {"provider": "gemini", "model": "invalid-model"},  # Will fail
        "gpt-4o"  # Fallback to this
    ]
)
```

### Monitoring

Logs show detailed fallback behavior:
```
[LLMClient] Starting chat completion with chain: [{'provider': 'gemini', 'model': 'gemini-2.5-flash'}, ...]
[LLMClient] Provider gemini (model: gemini-2.5-flash) failed: API error. Falling back to next in chain...
[LLMClient] Successfully completed with provider: openai, model: gpt-4o
```
