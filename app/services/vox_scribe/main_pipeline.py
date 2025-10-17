import os
import json
from dotenv import load_dotenv

from audio_preprocessing_service import AudioPreprocessor
from quadrant_service import Quadrant_service
from transcription_diarization_service import TranscriptionDiarizationService
from speaker_assignment_service import SpeakerAssignmentService

def main():
    """Main pipeline to run the full speaker identification and transcription process."""
    created_files = []
    try:
        print("--- Initializing Services ---")
        load_dotenv()
        preprocessor = AudioPreprocessor()
        qdrant_service = Quadrant_service()
        td_service = TranscriptionDiarizationService()
        assignment_service = SpeakerAssignmentService(qdrant_service)
        
        print("\n--- Processing Meeting Audio ---")
        meeting_audio_path = "audio_meeting_test_1.wav"
        if not os.path.exists(meeting_audio_path): raise FileNotFoundError(f"Missing meeting file: {meeting_audio_path}")
        
        # --- NEW: Define the number of speakers in the meeting audio ---
        # Set to 0 to let the model auto-detect.
        KNOWN_NUMBER_OF_SPEAKERS = 0
            
        clean_meeting_path = "clean_meeting_audio.wav"
        created_files.append(clean_meeting_path)
        preprocessor.process(input_path=meeting_audio_path, output_path=clean_meeting_path)

        transcription_segments = td_service.transcribe(meeting_audio_path) # Use raw audio for transcription
        
        # --- MODIFIED: Pass the number of speakers to the diarize method ---
        diarization_segments = td_service.diarize(meeting_audio_path, num_speakers=KNOWN_NUMBER_OF_SPEAKERS)
        
        speaker_name_map = assignment_service.identify_speakers(clean_meeting_path, diarization_segments)
        
        conversation_anonymous = assignment_service.assign_speakers_to_text(diarization_segments, transcription_segments)

        final_conversation = []
        for entry in conversation_anonymous:
            anonymous_speaker = entry["speaker"]
            identity = speaker_name_map.get(anonymous_speaker, {"name": "Unknown", "score": 0.0})
            
            final_conversation.append({
                "start_time": round(entry["start"], 2),
                "end_time": round(entry["end"], 2),
                "speaker": identity["name"],
                "text": entry["text"],
                "identification_score": round(identity.get("score", 0.0), 4)
            })

        print("\n\n=============================================")
        print("          Final Meeting Transcript           ")
        print("=============================================")
        if final_conversation:
            print(json.dumps(final_conversation, indent=2))
        else:
            print("⚠️ Could not generate a conversation transcript.")
            
    except Exception as e:
        print(f"🚨 A fatal error occurred in the main pipeline: {e}")
    
    finally:
        print("\n--- Cleaning up temporary processed files ---")
        for f in created_files:
            if os.path.exists(f):
                os.remove(f)
        print("Cleanup complete.")

if __name__ == "__main__":
    main()