import os
import uuid
from dotenv import load_dotenv

# --- UPDATED: Import the pre-processing service for the test block ---
from audio_preprocessing_service import AudioPreprocessor
from embedding_service import Embedding_service
from qdrant_client import QdrantClient, models

load_dotenv()

class Quadrant_service:
    def __init__(self):
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Optional
        
        if not self.QDRANT_URL or not self.QDRANT_COLLECTION_NAME:
            raise ValueError("QDRANT_URL and QDRANT_COLLECTION_NAME must be set in .env file.")
            
        self.client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        
        # This service now only depends on the embedding service
        self.embedding_service = Embedding_service()
        # --- REMOVED: The preprocessor is no longer part of this class ---

    def recreate_collection(self):
        """Deletes and recreates the collection to ensure a clean state."""
        try:
            self.client.delete_collection(collection_name=self.QDRANT_COLLECTION_NAME)
            print(f"✅ Removed existing collection '{self.QDRANT_COLLECTION_NAME}'.")
        except Exception:
            print(f"Collection '{self.QDRANT_COLLECTION_NAME}' did not exist. Creating new one.")
            pass
        
        try:
            self.client.create_collection(
                collection_name=self.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=192, distance=models.Distance.COSINE)
            )
            print(f"✅ Qdrant collection '{self.QDRANT_COLLECTION_NAME}' created.")
        except Exception as e:
            print(f"⚠️ Qdrant create error: {e}")
    
    def upsert(self, audio_path: str, speaker_name: str, metadata: dict = None):
        """Creates an embedding FROM THE PROVIDED AUDIO PATH and uploads it."""
        # --- REMOVED: Pre-processing logic is now handled outside this service. ---
        # This method now expects a path to a clean audio file.
        embedding = self.embedding_service.create_embedding(audio_path)
        if embedding is None:
            return None

        point_id = str(uuid.uuid4())
        payload = {"speaker_name": speaker_name, **(metadata or {})}

        self.client.upsert(
            collection_name=self.QDRANT_COLLECTION_NAME,
            points=[models.PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)],
            wait=True
        )

        print(f"✅ Upserted voice embedding for '{speaker_name}' from '{os.path.basename(audio_path)}'")
        return point_id

    def search_similar_voice(self, audio_path: str, top_k: int = 1):
        """Searches the collection using an embedding FROM THE PROVIDED AUDIO PATH."""
        print(f"\n🔎 Searching for speaker in '{os.path.basename(audio_path)}'...")
        # --- REMOVED: Pre-processing logic is now handled outside this service. ---
        # This method now expects a path to a clean audio file.
        query_embedding = self.embedding_service.create_embedding(audio_path)
        if query_embedding is None:
            return []

        results = self.client.search(
            collection_name=self.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True 
        )

        if not results:
            print("  -> No matches found.")
        
        for match in results:
            name = match.payload.get("speaker_name", "Unknown")
            score = match.score
            print(f"  🎙️ Top Match: '{name}' | Similarity Score: {score:.4f}")

        return results
    
    def get_all_voice_signatures(self, limit=100):
        # This method remains unchanged
        try:
            points, _ = self.client.scroll(
                collection_name=self.QDRANT_COLLECTION_NAME,
                limit=limit, with_payload=True, with_vectors=False
            )
            print(f"📦 Found {len(points)} points in Qdrant collection.")
            return points
        except Exception as e:
            print(f"❌ Error fetching points: {e}")
            return []

# ===================================================================================
#  TESTING BLOCK (Updated to show the separated workflow)
# ===================================================================================
if __name__ == '__main__':
    created_files = []
    try:
        load_dotenv()
        
        # --- NEW: Instantiate all required services for the test ---
        preprocessor = AudioPreprocessor()
        qdrant_service = Quadrant_service()
        
        print("\n--- Test Step 1: Recreating Qdrant Collection ---")
        qdrant_service.recreate_collection()
        
        print("\n--- Test Step 2: Enrolling Speakers (with explicit pre-processing) ---")
        enrollment_map = {
            "Guru": "audio_guru_2.wav",
            "Aniverthy": "audio_aniverthy_3.wav",
            "Gulshan": "audio_gulshan.wav"
        }
        for name, raw_path in enrollment_map.items():
            if not os.path.exists(raw_path):
                print(f"❌ Error: Enrollment file not found: '{raw_path}'.")
                exit()
            
            # 1. Pre-process the raw audio file
            clean_path = f"clean_{raw_path}"
            created_files.append(clean_path)
            preprocessor.process(input_path=raw_path, output_path=clean_path)
            
            # 2. Upsert using the CLEAN audio file
            qdrant_service.upsert(audio_path=clean_path, speaker_name=name)

        print("\n--- Test Step 3: Verifying Enrollment ---")
        all_points = qdrant_service.get_all_voice_signatures()
        assert len(all_points) == 3, f"Expected 3, found {len(all_points)}."
        print("✅ Verification successful: 3 speakers are enrolled.")

        print("\n--- Test Step 4: Performing Recognition Test (with explicit pre-processing) ---")
        unseen_raw_path = "audio_aniverthy_2.wav" 
        
        if not os.path.exists(unseen_raw_path):
            print(f"❌ Error: Search file not found: '{unseen_raw_path}'.")
            exit()

        # 1. Pre-process the unseen raw audio file
        unseen_clean_path = f"clean_{unseen_raw_path}"
        created_files.append(unseen_clean_path)
        preprocessor.process(input_path=unseen_raw_path, output_path=unseen_clean_path)
        
        # 2. Search using the CLEAN audio file
        search_results = qdrant_service.search_similar_voice(audio_path=unseen_clean_path)
        
        assert len(search_results) > 0, "Search should return a result."
        top_match = search_results[0]
        identified_name = top_match.payload.get("speaker_name")
        
        assert identified_name == "Aniverthy", f"Expected 'Aniverthy', but got '{identified_name}'."
        print(f"✅ Recognition successful! Top match is '{identified_name}' with a score of {top_match.score:.4f}.")
        print("\n🎉🎉🎉 End-to-end test completed successfully! 🎉🎉🎉")

    except Exception as e:
        print(f"\n❌ An error occurred during the test: {e}")

    finally:
        # --- NEW: Cleanup for all created clean files ---
        print("\n--- Cleaning up temporary processed files ---")
        for f in created_files:
            if os.path.exists(f):
                os.remove(f)
        print("Cleanup complete.")