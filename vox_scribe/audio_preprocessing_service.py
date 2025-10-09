# audio_preprocessing_service.py

import os
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
import torch
import torchaudio

class AudioPreprocessor:
    """A service to clean and standardize audio files for voice processing."""

    def __init__(self, target_lufs: float = -20.0, low_pass_hz: int = 8000, high_pass_hz: int = 100):
        """
        Initializes the preprocessor with target audio parameters.
        """
        self.target_lufs = target_lufs
        self.low_pass_hz = low_pass_hz
        self.high_pass_hz = high_pass_hz
        print("AudioPreprocessor initialized.")

    def _apply_band_pass_filter(self, sound: AudioSegment) -> AudioSegment:
        """Applies a high-pass and low-pass filter to the audio."""
        print("  - Applying band-pass filter...")
        sound = sound.high_pass_filter(self.high_pass_hz)
        sound = sound.low_pass_filter(self.low_pass_hz)
        return sound

    # --- THIS IS THE UPDATED METHOD ---
    def _apply_noise_reduction(self, sound: AudioSegment) -> AudioSegment:
        """Reduces background noise using a noise profile from the start of the audio."""
        print("  - Applying targeted noise reduction...")
        
        samples = np.array(sound.get_array_of_samples())
        sample_rate = sound.frame_rate

        # Ensure we handle different audio bit depths correctly
        if samples.dtype not in [np.int16, np.int32]:
            raise TypeError(f"Unsupported sample format: {samples.dtype}")
        
        max_val = np.iinfo(samples.dtype).max
        samples_float = samples.astype(np.float32) / max_val

        # --- NEW: Extract a noise profile from the start of the audio ---
        # This assumes the first 200ms contains representative background noise.
        noise_sample_duration_ms = 200
        noise_sample_size = int(sample_rate * (noise_sample_duration_ms / 1000.0))
        noise_clip = samples_float[:noise_sample_size]

        # --- NEW: Use the specific noise clip in the reduction function ---
        reduced_noise_float = nr.reduce_noise(
            y=samples_float, 
            y_noise=noise_clip, 
            sr=sample_rate
        )

        # Convert back to original integer type
        reduced_noise_int = (reduced_noise_float * max_val).astype(samples.dtype)

        return AudioSegment(
            reduced_noise_int.tobytes(),
            frame_rate=sample_rate,
            sample_width=sound.sample_width,
            channels=sound.channels
        )

    def _apply_loudness_normalization(self, sound: AudioSegment) -> AudioSegment:
        """Normalizes the audio to a target LUFS level."""
        print("  - Applying loudness normalization...")
        change_in_lufs = self.target_lufs - sound.dBFS
        return sound.apply_gain(change_in_lufs)

    def process(self, input_path: str, output_path: str):
        """
        Runs a full pre-processing pipeline on an audio file.
        """
        if not os.path.exists(input_path):
            print(f"❌ Error: Input file not found at '{input_path}'")
            return

        print(f"\nProcessing audio file: {input_path}")
        try:
            sound = AudioSegment.from_file(input_path)
            sound = sound.set_channels(1) # Ensure mono channel

            # Pipeline Order: Filter -> Noise Reduce -> Normalize
            # sound = self._apply_band_pass_filter(sound)
            # sound = self._apply_noise_reduction(sound)
            # sound = self._apply_loudness_normalization(sound)

            sound.export(output_path, format="wav")
            print(f"✅ Successfully processed audio saved to: {output_path}")

        except Exception as e:
            print(f"❌ An error occurred during audio processing: {e}")


# ===================================================================================
#  TESTING BLOCK
# ===================================================================================
if __name__ == '__main__':
    print("--- Testing AudioPreprocessor Service ---")

    TEST_INPUT_PATH = "audio_aniverthy_3.wav"
    DEBUG_DIR = "."
    TEST_OUTPUT_PATH = os.path.join(DEBUG_DIR, "audio_aniverthy_3_clean.wav") # Corrected typo in filename
    
    if not os.path.exists(TEST_INPUT_PATH):
        print(f"❌ Error: Test file not found at '{TEST_INPUT_PATH}'.")
        print("Please place the audio file in the same directory as the script.")
    else:
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            print(f"Input file: '{TEST_INPUT_PATH}'")
            print(f"Output will be saved to: '{TEST_OUTPUT_PATH}'")

            preprocessor = AudioPreprocessor()
            preprocessor.process(input_path=TEST_INPUT_PATH, output_path=TEST_OUTPUT_PATH)

            assert os.path.exists(TEST_OUTPUT_PATH), "Output file was not created."
            print(f"\n✅ Test complete. Check '{TEST_OUTPUT_PATH}' to hear the difference.")

        except Exception as e:
            print(f"🚨 Test failed with an error: {e}")