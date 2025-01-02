import os
import requests
import tempfile
import logging
from groq import Groq

logger = logging.getLogger(__name__)

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
VERSION = os.getenv("VERSION", "v20.0")  # Your Graph API version
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


def fetch_whatsapp_media_url(media_id):
    """
    Given a media_id, returns the direct download URL from WhatsApp Cloud API.
    """
    url = f"https://graph.facebook.com/{VERSION}/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data["url"]
    except requests.RequestException as e:
        logger.error(f"Error fetching WhatsApp media URL: {e}")
        raise


def download_media_as_bytes(media_url):
    """
    Downloads media bytes from the given media_url, returning bytes.
    """
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    try:
        response = requests.get(media_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logger.error(f"Error downloading WhatsApp media from {media_url}: {e}")
        raise


def transcribe_audio_from_whatsapp(media_id, mime_type, sha256):
    """
    1) Fetch the download URL from WhatsApp Cloud API
    2) Download the audio bytes
    3) Transcribe it using Groq (whisper-large-v3-turbo)
    4) Return the transcribed text
    """
    try:
        # 1) Fetch download URL
        media_url = fetch_whatsapp_media_url(media_id)
        # 2) Download audio bytes
        audio_data = download_media_as_bytes(media_url)
    except Exception as e:
        logger.error(f"Could not get audio data for media_id={media_id}: {e}")
        # Decide if you want to re-raise or return a fallback text
        raise

    # 3) Transcribe using Groq. We'll use a temporary file to hold audio.
    #    NamedTemporaryFile with delete=False is safer cross-platform (especially on Windows).
    #    We remove it manually once transcription is done.
    transcribed_text = ""
    tmp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name  # keep path for re-opening & later cleanup

        with open(tmp_file_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(tmp_file_path, f.read()),
                model="whisper-large-v3-turbo",
                response_format="json",
            )
            transcribed_text = transcription.text
    except Exception as e:
        logger.error(f"Groq transcription error: {e}")
        raise
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except OSError as e:
                logger.warning(f"Failed to remove temp file {tmp_file_path}: {e}")

    return transcribed_text
