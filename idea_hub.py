import whisper
import sounddevice as sd
import numpy as np

# Settings
SAMPLE_RATE = 44100  # Standard audio quality
DURATION = 3  # Adjust as needed

def record_audio(duration, sample_rate):
    """Records audio from the microphone and returns it as a NumPy array."""
    print("Recording... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio_data

# Load Whisper model
model = whisper.load_model("base")  # Can use "tiny", "small", "medium", or "large"

# Record and process audio
audio = record_audio(DURATION, SAMPLE_RATE)

# Convert NumPy array to Whisper-compatible format
audio = np.squeeze(audio)  # Remove unnecessary dimensions
audio = whisper.pad_or_trim(audio)  # Ensure correct input length
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Transcribe directly from memory
options = whisper.DecodingOptions(fp16=False, language="en")
result = whisper.decode(model, mel, options)

print("Transcription:", result.text)
