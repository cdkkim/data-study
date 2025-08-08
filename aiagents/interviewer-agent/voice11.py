
import os
import uuid
import requests
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings


API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in environment")

client = ElevenLabs(api_key=API_KEY)

def text_to_speech_file(
    text: str,
    voice_id: str,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    voice_settings: VoiceSettings | None = None
) -> str:
    """
    Converts text to speech and writes to a local file.
    Returns the output file path.
    """
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        output_format=output_format,
        voice_settings=voice_settings
    )
    filename = f"{uuid.uuid4()}.mp3"
    with open(filename, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    print(f"Saved audio to {filename}")
    return filename

if __name__ == "__main__":
    # https://elevenlabs.io/docs/conversational-ai/best-practices/conversational-voice-design#voices
    # Replace with a valid voice_id from your account
    VOICE_ID = 'kdmDKE6EkgrWrrykO9Qt' # "21m00Tcm4TlvDq8ikWAM"
    settings = VoiceSettings(stability=0.5, similarity_boost=0.8, style=0.0, use_speaker_boost=True)
    text = "안녕하세요! 오늘 날씨가 좋네요."
    filepath = text_to_speech_file(text, VOICE_ID, voice_settings=settings)
