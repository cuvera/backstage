import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseAnalysisAgent
from .agenda_agent import AgendaAgent
from .summary_agent import SummaryAgent
from .key_points_agent import KeyPointsAgent
from .decision_agent import DecisionAgent
from .action_item_agent import ActionItemAgent
from .scoring_agent import CallScoringAgent

logger = logging.getLogger(__name__)

class CallAnalysisCoordinator:
    """Orchestrates the specialized agents to analyze a meeting."""

    def __init__(self):
        self.agenda_agent = AgendaAgent()
        self.summary_agent = SummaryAgent()
        self.key_points_agent = KeyPointsAgent()
        self.decision_agent = DecisionAgent()
        self.action_item_agent = ActionItemAgent()
        self.scoring_agent = CallScoringAgent()
        self.base = BaseAnalysisAgent()

    async def analyze_meeting(
        self, 
        meeting_id: str, 
        tenant_id: str, 
        v2_transcript: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs the multi-agent analysis flow using V2 structured data with internal retry logic."""
        logger.info(f"[CallAnalysisCoordinator] Starting analysis for meeting {meeting_id}")
        
        MAX_RETRIES = 2
        RETRY_DELAY = 5
        
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    logger.info(
                        f"[CallAnalysisCoordinator] Retrying analysis "
                        f"(attempt {attempt + 1}/{MAX_RETRIES}) for {meeting_id}"
                    )
                    await asyncio.sleep(RETRY_DELAY)

                # 1. Prepare Diverse Inputs
                full_transcript = self._prepare_full_transcript(v2_transcript)
                topic_grouped_transcript = self._prepare_topic_grouped_transcript(v2_transcript)
                decisions_only = self._prepare_filtered_transcript(v2_transcript, ["decision"])
                actions_only = self._prepare_filtered_transcript(v2_transcript, ["actionable_item"])
                insights_only = self._prepare_filtered_transcript(v2_transcript, ["key_insight"])
                
                meta_str = json.dumps(metadata or {}, indent=2)

                # 2. Sequential Step: Agenda Inference
                identified_agenda = await self.agenda_agent.infer_agenda(full_transcript, meta_str)
                logger.info(f"[CallAnalysisCoordinator] Agenda inferred: {identified_agenda}")

                # 3. Parallel Step: Targeted analysis
                results = await asyncio.gather(
                    self.summary_agent.summarize(full_transcript, meta_str),
                    self.key_points_agent.extract_key_points(insights_only or topic_grouped_transcript, meta_str),
                    self.decision_agent.extract_decisions(decisions_only or full_transcript, meta_str),
                    self.action_item_agent.extract_action_items(actions_only or full_transcript, meta_str),
                    self.scoring_agent.score_call(full_transcript, meta_str, identified_agenda),
                    return_exceptions=False
                )

                # 4. Process Parallel Results
                # Exceptions are propagated so we can unpack directly
                summary, key_points, decisions, action_items, scoring = results

                # 5. Computational Steps (Non-LLM)
                talk_time_stats = self._calculate_talk_time_stats(v2_transcript)
                sentiment_overview = self._aggregate_sentiment(v2_transcript)

                # 6. Assemble Final Object
                analysis_result = {
                    "session_id": meeting_id,
                    "tenant_id": tenant_id,
                    "summary": summary,
                    "key_points": key_points,
                    "decisions": decisions,
                    "action_items": action_items,
                    "talk_time_stats": talk_time_stats,
                    "sentiment_overview": sentiment_overview,
                    "call_scoring": {
                        "score": scoring.get("score", 0.0),
                        "identified_agenda": identified_agenda,
                        "agenda_deviation_score": scoring.get("agenda_deviation_score", 0.0),
                        "action_item_completeness_score": scoring.get("action_item_completeness_score", 0.0),
                        "owner_clarity_score": scoring.get("owner_clarity_score", 0.0),
                        "due_date_quality_score": scoring.get("due_date_quality_score", 0.0),
                        "meeting_structure_score": scoring.get("meeting_structure_score", 0.0),
                        "signal_noise_ratio_score": scoring.get("signal_noise_ratio_score", 0.0),
                        "time_management_score": scoring.get("time_management_score", 0.0),
                        "positive_aspects": scoring.get("positive_aspects", []),
                        "areas_for_improvement": scoring.get("areas_for_improvement", [])
                    },
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "metadata": metadata or {}
                }

                logger.info(f"[CallAnalysisCoordinator] Analysis completed for meeting {meeting_id}")
                return analysis_result

            except Exception as e:
                last_error = e
                logger.error(f"[CallAnalysisCoordinator] Analysis failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"[CallAnalysisCoordinator] All retry attempts failed for {meeting_id}")
                    raise last_error

        # Should not reach here if loop is correct, but safe fallback
        raise last_error

    def _prepare_full_transcript(self, v2_data: Dict[str, Any]) -> str:
        """Flattens all V2 clusters into a single readable block."""
        lines = []
        for cluster in v2_data.get("segments", []):
            for transcription in cluster.get("transcriptions", []):
                speaker = transcription.get("speaker", "Unknown")
                start = transcription.get("start", "00:00")
                text = transcription.get("text", "")
                lines.append(f"[{start}] {speaker}: {text}")
        return "\n".join(lines)

    def _prepare_topic_grouped_transcript(self, v2_data: Dict[str, Any]) -> str:
        """Groups transcriptions by their V2 topics to help the KeyPointsAgent."""
        blocks = []
        for cluster in v2_data.get("segments", []):
            topics = ", ".join(cluster.get("topic", ["General"]))
            ctype = cluster.get("type", "discussion")
            
            blocks.append(f"--- TOPIC: {topics} (Type: {ctype}) ---")
            for tx in cluster.get("transcriptions", []):
                blocks.append(f"[{tx.get('start')}] {tx.get('speaker')}: {tx.get('text')}")
            blocks.append("")
        return "\n".join(blocks)

    def _prepare_filtered_transcript(self, v2_data: Dict[str, Any], target_types: List[str]) -> str:
        """Filters for specific cluster types (e.g., 'decision') for specialized agents."""
        lines = []
        for cluster in v2_data.get("segments", []):
            if cluster.get("type") in target_types:
                for tx in cluster.get("transcriptions", []):
                    speaker = tx.get("speaker", "Unknown")
                    start = tx.get("start", "00:00")
                    text = tx.get("text", "")
                    lines.append(f"[{start}] {speaker}: {text}")
        return "\n".join(lines)

    def _calculate_talk_time_stats(self, v2_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculates speaking duration per speaker from V2 clusters."""
        speaker_seconds = {}
        total_seconds = 0
        all_transcriptions = []
        
        for cluster in v2_data.get("segments", []):
            for tx in cluster.get("transcriptions", []):
                all_transcriptions.append(tx)
                speaker = tx.get("speaker", "Unknown")
                start = self.base._hhmmss_to_seconds(tx.get("start", "00:00"))
                end = self.base._hhmmss_to_seconds(tx.get("end", "00:00"))
                duration = max(0, end - start)
                
                speaker_seconds[speaker] = speaker_seconds.get(speaker, 0) + duration
                total_seconds += duration

        stats = []
        for speaker, seconds in speaker_seconds.items():
            share = round((seconds / total_seconds * 100), 1) if total_seconds > 0 else 0
            stats.append({
                "speaker": speaker,
                "total_duration": self.base._mmss_to_hhmmss(f"{int(seconds // 60)}:{int(seconds % 60)}"),
                "share_percent": share,
                "turns": sum(1 for tx in all_transcriptions if tx.get("speaker") == speaker)
            })
        return stats

    def _aggregate_sentiment(self, v2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregates sentiment from V2 transcript segments."""
        speaker_sentiments = {}
        all_sentiments = []
        
        for cluster in v2_data.get("segments", []):
            for tx in cluster.get("transcriptions", []):
                speaker = tx.get("speaker", "Unknown")
                sentiment = tx.get("sentiment", "neutral").lower()
                if speaker not in speaker_sentiments:
                    speaker_sentiments[speaker] = []
                speaker_sentiments[speaker].append(sentiment)
                all_sentiments.append(sentiment)

        def get_majority(sentiments):
            if not sentiments: return "neutral"
            counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
            for s in sentiments: 
                if s in counts: counts[s] += 1
            return max(counts, key=counts.get)

        per_speaker = {name: get_majority(sents) for name, sents in speaker_sentiments.items()}
        overall = get_majority(all_sentiments)
        
        return {
            "overall": overall,
            "per_speaker": per_speaker
        }
