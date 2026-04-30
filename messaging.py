"""Shared WhatsApp messaging helpers (Meta Cloud API).

Centralised so that both the webhook handler (api.py) and agent tools (agent.py)
can send WhatsApp messages and images without duplicating setup.
"""
import os
import requests

META_WA_PHONE_NUMBER_ID = os.environ.get("META_WA_PHONE_NUMBER_ID", "").strip()
META_WA_ACCESS_TOKEN = os.environ.get("META_WA_ACCESS_TOKEN", "").strip()

def configured() -> bool:
    return bool(META_WA_PHONE_NUMBER_ID and META_WA_ACCESS_TOKEN)

def _clean_number(number: str) -> str:
    """Remove 'whatsapp:' prefix and '+' from phone number."""
    number = number.replace("whatsapp:", "").replace("+", "")
    return number

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

def send_text(to_number: str, text: str, phone_number_id: str = "") -> bool:
    if not META_WA_ACCESS_TOKEN:
        print("⚠️  Meta WhatsApp API not configured (missing token).")
        return False
    
    sender_id = phone_number_id or META_WA_PHONE_NUMBER_ID
    if not sender_id:
        print("⚠️  No Phone Number ID available.")
        return False

    clean_to = _clean_number(to_number)
    url = f"https://graph.facebook.com/v18.0/{sender_id}/messages"
    headers = {
        "Authorization": f"Bearer {META_WA_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    ok = True
    for chunk in chunk_text(text):
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_to,
            "type": "text",
            "text": {"preview_url": False, "body": chunk}
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            r.raise_for_status()
        except Exception as e:
            err_text = getattr(e.response, 'text', '') if hasattr(e, 'response') else str(e)
            print(f"Failed to send WhatsApp text to {to_number} (from {sender_id}): {err_text}")
            ok = False
    return ok

def send_image(to_number: str, image_url: str, caption: str = "", phone_number_id: str = "") -> bool:
    """Send an image (or other media) via Meta WhatsApp API.
    The image_url MUST be a publicly reachable HTTPS URL.
    """
    if not META_WA_ACCESS_TOKEN:
        print("⚠️  Meta WhatsApp API not configured (missing token).")
        return False
    
    sender_id = phone_number_id or META_WA_PHONE_NUMBER_ID
    if not sender_id:
        print("⚠️  No Phone Number ID available.")
        return False

    clean_to = _clean_number(to_number)
    url = f"https://graph.facebook.com/v18.0/{sender_id}/messages"
    headers = {
        "Authorization": f"Bearer {META_WA_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": clean_to,
        "type": "image",
        "image": {
            "link": image_url
        }
    }
    if caption:
        payload["image"]["caption"] = caption
        
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        err_text = getattr(e.response, 'text', '') if hasattr(e, 'response') else str(e)
        print(f"Failed to send WhatsApp image to {to_number} (from {sender_id}): {err_text}")
        return False
