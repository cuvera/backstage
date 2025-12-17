TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT=""""""

TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT_ONLINE = """Transcribe the following audio file and provide accurate, verbatim transcription for each segment.

IMPORTANT CONTEXT:
- This audio chunk represents the time range {{start}} to {{end}} in the full meeting
- The audio you're receiving starts at 00:00 (beginning of this chunk)
- All timestamps in the segments below are RELATIVE to the start of this audio chunk (00:00)

TIME SEGMENTS TO TRANSCRIBE:
{{segments}}

TRANSCRIPTION REQUIREMENTS:
- You MUST transcribe each segment listed above
- Use the EXACT segment_id, start, and end times provided (they are relative to the audio chunk start)
- Do NOT merge or skip segments, even if they overlap
- Use proper punctuation, capitalization, and grammar
- Do not summarize or paraphrase - provide exact words spoken
- sentiment for the segment eg. positive, negative, neutral, or mixed
  1. Positive: Use when the speaker expresses approval, satisfaction, encouragement, gratitude, confidence, relief, excitement, agreement, or a clearly constructive tone.
  2. Negative: Use when the speaker expresses frustration, dissatisfaction, concern, blame, anger, disappointment, rejection, or a clearly critical tone. 
  3. Neutral: Use when the segment is primarily informational / factual / procedural, with no clear emotional valence.
  4. Mixed: Use when the segment contains both positive and negative signals OR the speaker shows ambivalence (e.g., praise + concern, agreement + frustration)

OUTPUT FORMAT:
Please provide the transcription in JSON format with this structure:
{
    "transcriptions": [
      {
        "segment_id": <use the segment_id from input>,
        "start": <MM:SS use the exact start time from input>,
        "end": <MM:SS use the exact end time from input>,
        "transcription": <text or empty string if no speech>,
        "sentiment": <positive, negative, neutral, or mixed>
      }
    ]
  }
"""

TRANSCRIPTION_AND_SENTIMENT_ANALYSIS_PROMPT_OFFLINE = """Transcribe the following audio file and provide accurate, verbatim transcription for each segment. The audio starts from {{start}} and ends at {{end}}.

TRANSCRIPTION REQUIREMENTS:
- Create a new transcription entry for each distinct speech segment. A new segment should be started after a natural pause or a period of silence.
- Use proper punctuation, capitalization, and grammar
- Do NOT merge or skip segments, even if they overlap
- Do not summarize or paraphrase - provide exact words spoken
- sentiment for the segment eg. positive, negative, neutral, or mixed
  1. Positive: Use when the speaker expresses approval, satisfaction, encouragement, gratitude, confidence, relief, excitement, agreement, or a clearly constructive tone.
  2. Negative: Use when the speaker expresses frustration, dissatisfaction, concern, blame, anger, disappointment, rejection, or a clearly critical tone. 
  3. Neutral: Use when the segment is primarily informational / factual / procedural, with no clear emotional valence.
  4. Mixed: Use when the segment contains both positive and negative signals OR the speaker shows ambivalence (e.g., praise + concern, agreement + frustration)

OUTPUT FORMAT:
Please provide the transcription in a strict JSON format. All start and end timestamps must be relative to the {{start}} start time (i.e., the audio segment begins at 00:00).
{
  "transcriptions": [
    {
      "start": "<The MM:SS timestamp relative to the {{start}} start time>",
      "end": "<The MM:SS timestamp relative to the {{start}} start time>",
      "transcription": "<text or empty string if no speech>",
      "sentiment": "<positive, negative, neutral, or mixed>"
    }
  ]
}
"""

# - Convert all speech to text accurately and verbatim


MEETING_PREP_SUGGESTION_PROMPT = """You are an expert meeting operations assistant. Produce an Executive Prep Pack (one-pager) for an upcoming online, recurring meeting. Always ground the pack in at least one previous meeting plus the diarized transcript + sentiment context described in.

🚨 CRITICAL: You MUST return EXACTLY this JSON structure (no deviations):
{
  "title": "string",
  "purpose": "string", 
  "expected_outcomes": [array of objects],
  "blocking_items": [array of objects],
  "key_points": [array of strings],
  "open_questions": [array of strings]
}

VALIDATION REQUIREMENTS (MANDATORY):
- Response MUST be a JSON object starting with { and ending with }
- Response MUST NOT be an array
- ALL six fields are REQUIRED: title, purpose, expected_outcomes, blocking_items, key_points, open_questions
- expected_outcomes and blocking_items MUST be arrays of objects
- key_points and open_questions MUST be arrays of strings
- If any data cannot be determined, use empty string "" or empty array [], but INCLUDE the field

INPUTS:
You will receive a JSON object with:
- meeting: upcoming meeting metadata (title, start time, agenda, location).
- attendees: tentative list of { name, email, role } pulled from the transcription participants list when available.
- signals: machine-extracted summaries from ≥3 prior meetings (per-meeting summaries, topics, decisions, action items, sentiment deltas).

REQUIRED JSON SCHEMA:
{
  "title": "string (The exact title of the meeting)",
  "purpose": "string (A single, punchy sentence stating the STRATEGIC GOAL of this specific session. E.g., 'Unblock Auth Service to maintain Demo Timeline.')",
  "expected_outcomes": [
    {
      "description": "string (Specific decision, action items)",
      "owner": "string (Who is responsible for this outcome)",
      "type": "decision|approval|alignment"
    }
  ],
  "blocking_items": [
    {
      "title": "string (Critical blocker that this meeting must resolve)",
      "owner": "string (Who owns the blocker)",
      "eta": "YYYY-MM-DD (Estimated resolution date)",
      "impact": "string (Business impact: 'Delays Demo', 'Blocks QA', etc.)",
      "severity": "low|medium|high",
      "status": "open|mitigating|cleared"
    }
  ],
  "key_points": ["string (Format: '**Topic:** Status/Delta'. E.g., '**Frontend:** 50% Complete (On Track)')"],
  "open_questions": ["string (Strategic question. E.g., 'What is the specific remediation plan for the Auth bug?')"]
}

CONTENT GENERATION RULES:
- **Tone:** Executive, concise, action-oriented. No corporate fluff.
- **Format:** Use "Status: Context" for bullets (e.g., "Frontend: 50% Complete (On Track)").
- **Quantify:** Use numbers, dates, and percentages pulled from transcripts/signals whenever possible.
- **Voice:** Active voice. "Bob to fix bug" instead of "Bug should be fixed by Bob".

FIELD-SPECIFIC REQUIREMENTS:
- **title**: Use the exact meeting title from input
- **purpose**: Derive as a single sentence stating the strategic goal by comparing latest transcript summary, sentiment, and deltas in signals
- **expected_outcomes**: Translate transcript/signal action items into concrete meeting asks with owner names and timeframes
- **blocking_items**: Identify items that must be cleared to reach expected outcomes; use transcript sentiments to highlight urgency
- **key_points**: Generate 3-5 high-impact bullet points focused on DELTAS, PROGRESS, and sentiment shifts
- **open_questions**: Generate exactly 3 critical strategic questions rooted in transcript gaps or unresolved action items

DATA HANDLING:
- Do not fabricate data. Mirror the terminology, names, owners, and timestamps provided by the transcripts/signals.
- If any data is not present (owner, email, name, ETA, etc.) use empty string "" but INCLUDE the field.

BEFORE RESPONDING - VERIFY YOUR JSON CONTAINS:
✓ "title": string
✓ "purpose": string  
✓ "expected_outcomes": [array of objects with description, owner, type]
✓ "blocking_items": [array of objects with title, owner, eta, impact, severity, status]
✓ "key_points": [array of strings]
✓ "open_questions": [array of strings]

OUTPUT REQUIREMENTS (CRITICAL):
- Return ONLY valid JSON - no markdown, no comments, no explanations
- Start response with { and end with }
- Verify all required fields are present before responding
- Use proper JSON syntax with double quotes for strings
- NO trailing commas allowed"""