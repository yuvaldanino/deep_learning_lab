import gradio as gr
import whisper
import tempfile
import numpy as np
import scipy.io.wavfile as wav

def transcribe_audio(audio):
    if audio is None:
        return "No audio recorded."

    sample_rate, audio_data = audio

    # Save as temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_wav.name, sample_rate, np.array(audio_data, dtype=np.int16))

    # Transcribe using Whisper
    model = whisper.load_model("base")  # Change to "small", "medium", etc., for different models
    result = model.transcribe(temp_wav.name)

    return result["text"]

# Gradio UI with audio input and a button to trigger transcription
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="numpy", label="Record your audio"),
    outputs="text",
    live=True
)

iface.launch()