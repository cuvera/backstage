ONLINE_TRANSCRIPTION = """Transcribe the following audio file and provide accurate, verbatim transcription for each segment.

KNOWN PARTICIPANTS:
{{participants}}

The audio contains segments corresponding to speaker turns based on Google Meet's speaker detection.
Use the known participants list above to help identify speakers when possible.

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

OFFLINE_TRANSCRIPTION = """Transcribe the following audio file and provide accurate, verbatim transcription for each segment. The audio starts from {{start}} and ends at {{end}}.

KNOWN PARTICIPANTS:
{{participants}}

The audio contains conversation that needs to be segmented based on natural pauses and speaker changes.
Use the known participants list above to help identify speakers when possible.

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

#- general_discussion: General conversation, casual exchanges, routine status updates, small talk, acknowledgments, or informational statements that do not constitute actionable items, decisions, key insights, or questions. Use this as the default classification for segments that maintain conversational flow but lack significant meeting content requiring tracking.
SEGMENT_CLASSIFICATION_PROMPT = """You are an expert meeting analyst. Your task is to analyze a meeting transcription to identify and group related conversational segments into chronological clusters. After grouping, you must classify each cluster according to the provided definitions.

CRITICAL INSTRUCTIONS:
    - Strict Chronological Order: Process the transcript from start to finish. The cluster_id must be assigned incrementally (1, 2, 3...) based on the timeline of the conversation. Do not group all items of the same type together; the output must reflect the natural flow of the meeting topics.
    - Comprehensive Segment Grouping: A single topic or conversational point is often discussed across multiple, consecutive transcript segments.
    - Identify the Core Point: Find the segment that introduces the main idea (e.g., asks the primary question, states the decision).
    - Include All Related Exchanges: You must group this core segment with all immediately following short responses, clarifications, agreements ("Okay", "Right", "Mhm"), and follow-up statements that directly relate to it.
    - Define Cluster Boundaries: A cluster ends when the conversation clearly pivots to a new, distinct topic. Do not leave any segments unclustered. Every single segment must belong to a cluster.
    - Accurate Classification: Assign one of the following types to each cluster based on its primary purpose. Read the definitions carefully.
    - Group Related Segment: If consecutive segment discuss the same action/decision/insight, group them

CLASSIFICATION DEFINITIONS:
    - actionable_item: A specific, delegable task or future commitment requiring explicit action. This includes assignments, follow-ups, and promises to do something.
    - decision: A finalized conclusion, agreement, or choice that resolves a discussion or confirms a plan. This is the point where a consensus is reached.
    - key_insight: A critical observation, strategic discovery, identified risk, concern, or significant piece of feedback. This captures important context or problems that are not direct tasks or decisions.
    - question: A formal request for information or clarification that identifies an unknown. This includes direct questions and segments that are clearly seeking input or answers.

ENRICHMENT FIELDS (for ALL Clusters):
    - topic: Identify topics discussed in the meeting

OUTPUT FORMAT (JSON):
    {
        "clusters": [
            {
            "cluster_id": <Integer starting from 1, following the strict chronological order of the meeting flow>,
            "type": <classification type for each cluster>,
            "topic": <Array of topics>,
            "segment_ids": [int]
            }
        ]
    }
"""