import torch
import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from typing import List, Dict, Any
import json
from app.core.config import settings

class TranscriptionDiarizationService:
    """Handles transcription and speaker diarization."""

    def __init__(self):
        self.HF_TOKEN = settings.HUGGINGFACE_TOKEN
        self.WHISPER_MODEL_NAME = settings.WHISPER_MODEL_NAME or "base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            print(f"Loading Whisper model: {self.WHISPER_MODEL_NAME}...")
            self.whisper_model = WhisperModel(self.WHISPER_MODEL_NAME, device=str(self.device).split(':')[0], compute_type="float32")

            print("Loading pyannote diarization pipeline...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                # "pyannote/speaker-diarization-3.1",
                token=self.HF_TOKEN
            ).to(self.device)
            print("✅ Transcription and Diarization models loaded.")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribes audio using the pre-loaded model."""
        try:
            print("\n🎙️ Starting transcription...")
            segments, _ = self.whisper_model.transcribe(audio_path, beam_size=5)
            results = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
            print("✅ Transcription complete.")
            return results
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return []

    def diarize(self, audio_path: str, num_speakers: int = 0) -> List[Dict[str, Any]]:
        """
        Performs speaker diarization using the pre-loaded model.

        Args:
            audio_path (str): Path to the audio file.
            num_speakers (int): The exact number of speakers in the audio.
                                If 0, the model will attempt to auto-detect.
        """
        try:
            print(f"\n🎧 Performing speaker diarization... (Known speakers: {'Auto' if num_speakers == 0 else num_speakers})")
            
            if num_speakers > 0:
                diarization = self.diarization_pipeline(audio_path, num_speakers=num_speakers)
            else:
                diarization = self.diarization_pipeline(audio_path)
            
            results = [{"speaker": speaker, "start": turn.start, "end": turn.end} 
                       for turn, speaker in diarization.speaker_diarization]
            print("✅ Diarization complete.")
            return results
        except Exception as e:
            print(f"❌ Error during diarization: {e}")
            return []


# ===================================================================================
#  TESTING BLOCK
# ===================================================================================
if __name__ == '__main__':
    print("--- Testing TranscriptionDiarizationService ---")

    TEST_AUDIO_PATH = "audio_conversation_with_4.wav"
    KNOWN_NUMBER_OF_SPEAKERS = 4

    # --- NEW: Helper function to merge results ---
    def assign_speakers_to_text(diarization_segments: List[Dict[str, Any]], transcription_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assigns speaker labels to each transcription segment based on overlap."""
        conversation = []
        for t_seg in transcription_segments:
            best_overlap, best_speaker = 0.0, "Unknown"
            for d_seg in diarization_segments:
                overlap = max(0, min(t_seg["end"], d_seg["end"]) - max(t_seg["start"], d_seg["start"]))
                if overlap > best_overlap:
                    best_overlap, best_speaker = overlap, d_seg["speaker"]
            
            conversation.append({
                "start_time": round(t_seg["start"], 2),
                "end_time": round(t_seg["end"], 2),
                "speaker": best_speaker,
                "text": t_seg["text"].strip()
            })
        return conversation

    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"❌ Error: Test audio file not found at '{TEST_AUDIO_PATH}'.")
    else:
        try:
            td_service = TranscriptionDiarizationService()

            transcription_results = td_service.transcribe(TEST_AUDIO_PATH)
            assert transcription_results, "Transcription failed."

            diarization_results = td_service.diarize(TEST_AUDIO_PATH, num_speakers=KNOWN_NUMBER_OF_SPEAKERS)
            assert diarization_results, "Diarization failed."
            
            # --- NEW: Merge the results ---
            print("\n🤝 Merging transcription and diarization results...")
            final_conversation = assign_speakers_to_text(diarization_results, transcription_results)
            print("✅ Merging complete.")

            # --- NEW: Dump the final JSON to a file ---
            output_filename = "conversation_with_speakers.json"
            with open(output_filename, "w") as f:
                json.dump(final_conversation, f, indent=2)
            print(f"✅ Final transcript with speaker labels saved to '{output_filename}'")
            
            # Print a summary to the console
            print("\n--- Final Conversation (Sample) ---")
            print(json.dumps(final_conversation, indent=2)) # Print first 5 entries for a quick look

            print("\n🎉🎉🎉 Service test completed successfully! 🎉🎉🎉")

        except Exception as e:
            print(f"🚨 A fatal error occurred during the test: {e}")