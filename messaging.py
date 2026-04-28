"""Shared WhatsApp messaging helpers (Twilio-backed).

Centralised so that both the webhook handler (api.py) and agent tools (agent.py)
can send WhatsApp messages and images without duplicating Twilio setup.
"""
import os
from twilio.rest import Client as TwilioClient

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
_raw_from = (os.environ.get("TWILIO_WHATSAPP_NUMBER") or "").strip()
if _raw_from and not _raw_from.startswith("whatsapp:"):
    _raw_from = "whatsapp:" + _raw_from
TWILIO_FROM = _raw_from
_twilio = TwilioClient(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN) else None


def configured() -> bool:
    return bool(_twilio and TWILIO_FROM)


def chunk_text(text: str, limit: int = 1500):
    text = text or ""
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < 200:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def send_text(to_number: str, text: str) -> bool:
    if not configured():
        print("⚠️  Twilio not configured.")
        return False
    ok = True
    for chunk in chunk_text(text):
        try:
            _twilio.messages.create(from_=TWILIO_FROM, to=to_number, body=chunk)
        except Exception as e:
            print(f"Failed to send WhatsApp text to {to_number}: {e}")
            ok = False
    return ok


def send_image(to_number: str, image_url: str, caption: str = "") -> bool:
    """Send an image (or other media) via Twilio WhatsApp.
    The image_url MUST be a publicly reachable HTTPS URL.
    """
    if not configured():
        print("⚠️  Twilio not configured.")
        return False
    try:
        _twilio.messages.create(
            from_=TWILIO_FROM,
            to=to_number,
            body=caption or "",
            media_url=[image_url],
        )
        return True
    except Exception as e:
        print(f"Failed to send WhatsApp image to {to_number}: {e}")
        return False
