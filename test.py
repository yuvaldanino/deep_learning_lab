import whisper
import sounddevice as sd
import numpy as np
import wave
import tempfile

# Settings
SAMPLE_RATE = 44100  # Standard audio quality
DURATION = 5  # Recording time in seconds

def record_audio(duration, sample_rate):
    """Records audio from the microphone and saves it to a temporary WAV file."""
    print("Recording... Speak now!")
    
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        with wave.open(temp_audio_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        return temp_audio_file.name  # Return the file path

# Load Whisper model
model = whisper.load_model("base")  # Try "small", "medium", or "large" for better accuracy

# Record and transcribe
audio_file = record_audio(DURATION, SAMPLE_RATE)

print("Transcribing...")
result = model.transcribe(audio_file, language="en")  # Force English
print("Transcription:", result["text"])
