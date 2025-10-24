import os
import numpy as np
import torchaudio
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# --- NEW: Import the assignment algorithm from scipy ---
from scipy.optimize import linear_sum_assignment

from app.services.vox_scribe.quadrant_service import Quadrant_service
from app.services.vox_scribe.audio_preprocessing_service import AudioPreprocessor
from app.services.vox_scribe.transcription_diarization_service import TranscriptionDiarizationService

load_dotenv()

class SpeakerAssignmentService:
    """Identifies speakers and assigns them to transcribed text."""

    def __init__(self, qdrant_service: Quadrant_service):
        self.qdrant_service = qdrant_service
        print("✅ SpeakerAssignmentService initialized.")
    
    # --- THIS IS THE REWRITTEN METHOD ---
    def identify_speakers(self, clean_audio_path: str, diarization_segments: List[Dict[str, Any]], similarity_threshold: float = 0.40) -> Dict[str, Dict[str, Any]]:
        """
        Identifies anonymous speakers using the Hungarian algorithm for optimal 1-to-1 assignment.
        """
        MIN_DURATION_SECONDS = 1.0
        speaker_map = {}
        try:
            waveform, sr = torchaudio.load(clean_audio_path)
            anonymous_speakers = sorted(list(set(seg["speaker"] for seg in diarization_segments)))
            print(f"🔍 Identifying anonymous speakers: {anonymous_speakers}")

            # --- Step 1: Build the Score Matrix ---
            # Get all enrolled speakers from the database
            enrolled_points = self.qdrant_service.get_all_voice_signatures()
            if not enrolled_points:
                print("⚠️ No enrolled speakers found in the database.")
                return {anon_speaker: {"name": f"Unknown_{anon_speaker.split('_')[-1]}", "score": 0.0} for anon_speaker in anonymous_speakers}

            enrolled_speakers = {point.payload['speaker_name']: np.array(self.qdrant_service.client.retrieve(collection_name=self.qdrant_service.QDRANT_COLLECTION_NAME, ids=[point.id], with_vectors=True)[0].vector) for point in enrolled_points}
            enrolled_names = list(enrolled_speakers.keys())
            
            score_matrix = np.zeros((len(anonymous_speakers), len(enrolled_names)))

            for i, anon_speaker in enumerate(anonymous_speakers):
                # Create the aggregated voiceprint for the anonymous speaker
                speaker_segments = [seg for seg in diarization_segments if seg["speaker"] == anon_speaker and (seg["end"] - seg["start"]) > MIN_DURATION_SECONDS]
                if not speaker_segments: continue

                embeddings = []
                for segment in speaker_segments:
                    start_frame, end_frame = int(segment["start"] * sr), int(segment["end"] * sr)
                    segment_waveform = waveform[:, start_frame:end_frame]
                    temp_chunk_path = "temp_segment_for_embedding.wav"
                    torchaudio.save(temp_chunk_path, segment_waveform, sr)
                    embedding = self.qdrant_service.embedding_service.create_embedding(temp_chunk_path)
                    if embedding is not None: embeddings.append(embedding)
                    os.remove(temp_chunk_path)
                
                if not embeddings: continue
                
                query_embedding = np.mean(embeddings, axis=0)
                query_embedding /= np.linalg.norm(query_embedding)

                # Calculate similarity score against EVERY enrolled speaker
                for j, name in enumerate(enrolled_names):
                    # Cosine similarity is the dot product of normalized vectors
                    score = np.dot(query_embedding, enrolled_speakers[name])
                    score_matrix[i, j] = score

            # --- Step 2: Find the Optimal Pairing ---
            # The algorithm finds the minimum cost, so we use 1 - score (or a large number - score)
            cost_matrix = 1 - score_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # --- Step 3: Apply the Assignment and Threshold ---
            print("\n--- Optimal Speaker Assignment ---")
            for i, j in zip(row_ind, col_ind):
                anon_speaker = anonymous_speakers[i]
                identified_name = enrolled_names[j]
                score = score_matrix[i, j]

                if score > similarity_threshold:
                    speaker_map[anon_speaker] = {"name": identified_name, "score": score}
                    print(f"  -> Mapped {anon_speaker} to '{identified_name}' with score {score:.4f}")
                else:
                    speaker_map[anon_speaker] = {"name": f"Unknown_{anon_speaker.split('_')[-1]}", "score": score}
                    print(f"  -> Match for {anon_speaker} to '{identified_name}' ({score:.4f}) was below threshold.")
            
            # Add any unassigned anonymous speakers as Unknown
            for i, anon_speaker in enumerate(anonymous_speakers):
                if anon_speaker not in speaker_map:
                     speaker_map[anon_speaker] = {"name": f"Unknown_{anon_speaker.split('_')[-1]}", "score": 0.0}

            return speaker_map
        except Exception as e:
            print(f"❌ Error during speaker identification: {e}")
            return {seg["speaker"]: {"name": "Unknown", "score": 0.0} for seg in diarization_segments}

    def assign_speakers_to_text(self, diarization_segments: List[Dict[str, Any]], transcription_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # This method remains the same
        conversation = []
        for t_seg in transcription_segments:
            best_overlap, best_speaker = 0.0, "Unknown"
            for d_seg in diarization_segments:
                overlap = max(0, min(t_seg["end"], d_seg["end"]) - max(t_seg["start"], d_seg["start"]))
                if overlap > best_overlap:
                    best_overlap, best_speaker = overlap, d_seg["speaker"]
            conversation.append({"start": t_seg["start"], "end": t_seg["end"], "speaker": best_speaker, "text": t_seg["text"].strip()})
        return conversation

# ===================================================================================
#  TESTING BLOCK (Updated to use live data from other services)
# ===================================================================================
if __name__ == '__main__':
    print("--- Testing SpeakerAssignmentService with Live Data ---")
    
    created_files = []
    try:
        # --- 1. Initialize all necessary services ---
        preprocessor = AudioPreprocessor()
        qdrant_service = Quadrant_service()
        td_service = TranscriptionDiarizationService()

        # --- 2. Enroll speakers ---
        print("\n--- Step 1: Enrolling speakers ---")
        qdrant_service.recreate_collection()
        enrollment_map = {
            "Guru": "audio_guru_2.wav",
            "Aniverthy": "audio_aniverthy_3.wav",
            "Gulshan": "audio_gulshan.wav" # Added Gulshan as requested
        }
        for name, raw_path in enrollment_map.items():
            if not os.path.exists(raw_path): raise FileNotFoundError(f"Missing enrollment file: {raw_path}")
            clean_path = f"clean_{raw_path}"
            created_files.append(clean_path)
            preprocessor.process(input_path=raw_path, output_path=clean_path)
            qdrant_service.upsert(audio_path=clean_path, speaker_name=name)

        # --- 3. Generate live data from the meeting file ---
        print("\n--- Step 2: Generating live data from meeting audio ---")
        meeting_audio_path = "audio_meeting_test_1.wav"
        if not os.path.exists(meeting_audio_path): raise FileNotFoundError(f"Missing meeting file: {meeting_audio_path}")
        
        # NOTE: We use the RAW audio for transcription and diarization, as decided.
        live_transcription = td_service.transcribe(meeting_audio_path)
        live_diarization = td_service.diarize(meeting_audio_path)
        
        assert live_transcription, "Live transcription failed."
        assert live_diarization, "Live diarization failed."
        print("✅ Live transcription and diarization data generated successfully.")

        # --- 4. Test the SpeakerAssignmentService with the live data ---
        print("\n--- Step 3: Testing SpeakerAssignmentService methods ---")
        assignment_service = SpeakerAssignmentService(qdrant_service)

        # A) Test identify_speakers
        # NOTE: We use the CLEAN audio for identification, for best results.
        clean_meeting_path = "clean_meeting_for_id.wav"
        created_files.append(clean_meeting_path)
        preprocessor.process(input_path=meeting_audio_path, output_path=clean_meeting_path)
        speaker_map = assignment_service.identify_speakers(clean_meeting_path, live_diarization)
        
        print("\n--- Speaker Identification Results ---")
        print(json.dumps(speaker_map, indent=2))
        assert speaker_map, "Speaker identification failed."

        # B) Test assign_speakers_to_text
        conversation_anonymous = assignment_service.assign_speakers_to_text(live_diarization, live_transcription)
        assert conversation_anonymous, "Speaker-to-text assignment failed."

        # C) Final verification
        final_conversation = []
        for entry in conversation_anonymous:
            entry['speaker'] = speaker_map.get(entry['speaker'], 'Unknown')
            final_conversation.append(entry)

        print("\n--- Final Named Conversation (Sample) ---")
        print(json.dumps(final_conversation[:5], indent=2)) # Print first 5 entries
        
        print("\n🎉🎉🎉 Service test completed successfully! 🎉🎉🎉")

    except Exception as e:
        print(f"🚨 A fatal error occurred during the test: {e}")
    
    finally:
        print("\n--- Cleaning up temporary processed files ---")
        for f in created_files:
            if os.path.exists(f):
                os.remove(f)
        print("Cleanup complete.")