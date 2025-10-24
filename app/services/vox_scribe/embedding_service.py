# embedding_service.py

import os
import numpy as np
import torch
import torchaudio  # <-- Now used for loading audio
from dotenv import load_dotenv
from typing import Optional, List
from scipy.spatial.distance import cdist

# --- NEW: Import the SpeechBrain model ---
from speechbrain.pretrained import EncoderClassifier

load_dotenv()

class Embedding_service:
    """A service to create high-quality, normalized speaker embeddings using SpeechBrain."""
    
    def __init__(self):
        """Initializes the service by loading the pre-trained SpeechBrain model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # --- UPDATED: Load the SpeechBrain ECAPA-TDNN model ---
            print("Loading SpeechBrain embedding model...")
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join("/tmp", "speechbrain_models"),
                run_opts={"device": device}
            )
            print("✅ SpeechBrain model loaded.")
        except Exception as e:
            print(f"❌ Could not load SpeechBrain model: {e}")
            raise
    
    def create_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Generates a L2-normalized speaker embedding from an audio file path."""
        try:
            # --- UPDATED: Logic for SpeechBrain model ---
            # 1. Load the audio file and get the waveform
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 2. Get the embedding from the model's encoder
            with torch.no_grad():
                embedding_tensor = self.model.encode_batch(waveform)
            
            # 3. Squeeze the tensor and convert it to a numpy array
            embedding = embedding_tensor.squeeze().cpu().numpy()
            
            # --- REMOVED: Manual L2 normalization is no longer needed ---
            
            print(f"✅ Embedding created for {os.path.basename(audio_path)} with shape: {embedding.shape}")
            return embedding

        except Exception as e:
            print(f"❌ Error generating embedding for {audio_path}: {e}")
            return None

    def _compare_embeddings(self, embeddings: List[np.ndarray], input_embedding: np.ndarray) -> List[float]:
        """
        [Internal] Compares a list of embeddings against a single input embedding.
        """
        if input_embedding is None or not embeddings:
            print("⚠️ Missing embeddings or input embedding.")
            return []

        distances = []
        inp = np.reshape(input_embedding, (1, -1))

        for idx, emb in enumerate(embeddings):
            emb = np.reshape(emb, (1, -1))
            distance = cdist(emb, inp, metric="cosine")[0, 0]
            distances.append(distance)
            print(f"🧩 Distance to embedding {idx + 1}: {distance:.4f} (Similarity: {1-distance:.4f})")

        return distances

# ===================================================================================
#  TESTING BLOCK (This block remains the same, but now uses the new model)
# ===================================================================================
if __name__ == '__main__':
    print("--- Testing Embedding Service with Realistic Scenario (using SpeechBrain) ---")

    speaker_a_enroll_path = "audio_aniverthy_3.wav"
    speaker_b_enroll_path = "audio_guru_2.wav"
    speaker_a_unseen_path = "audio_aniverthy_unseen.wav"
    
    required_files = [speaker_a_enroll_path, speaker_b_enroll_path, speaker_a_unseen_path]
    if not all(os.path.exists(f) for f in required_files):
        print("❌ Error: One or more required audio files are missing.")
        print(f"Please ensure '{speaker_a_enroll_path}', '{speaker_b_enroll_path}', and '{speaker_a_unseen_path}' are present.")
    else:
        try:
            embedding_service = Embedding_service()

            print("\n--- Step 1: Enrolling known speakers ---")
            embedding_aniverthy = embedding_service.create_embedding(speaker_a_enroll_path)
            embedding_guru = embedding_service.create_embedding(speaker_b_enroll_path)
            
            library_embeddings = [embedding_aniverthy, embedding_guru]
            library_names = ["Aniverthy", "Guru"]
            
            assert all(emb is not None for emb in library_embeddings), "Failed to create one or more enrollment embeddings."
            print("✅ Enrollment embeddings created successfully.")

            print("\n--- Step 2: Creating embedding for an unseen audio sample ---")
            unseen_embedding = embedding_service.create_embedding(speaker_a_unseen_path)
            assert unseen_embedding is not None, "Failed to create the unseen embedding."

            print(f"\n--- Step 3: Comparing '{os.path.basename(speaker_a_unseen_path)}' against the library ---")
            distances = embedding_service._compare_embeddings(library_embeddings, unseen_embedding)
            
            print("\n--- Verification ---")
            if distances:
                closest_index = np.argmin(distances)
                identified_speaker = library_names[closest_index]
                
                print(f"🔎 Closest match in the database is: '{identified_speaker}'")
                
                assert identified_speaker == "Aniverthy", f"Expected to identify 'Aniverthy', but got '{identified_speaker}'."
                print("✅ Recognition successful: The correct speaker was identified.")
                
                assert distances[0] < distances[1], "The distance to the correct speaker should be the smallest."
                print("✅ Differentiation successful: The unseen sample is closer to the correct speaker.")

        except Exception as e:
            print(f"🚨 Test failed with an error: {e}")