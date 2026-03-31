import os
import base64
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI(
    title="Visiting Card Parser API",
    description="Upload a visiting card image and get structured contact details as JSON",
    version="1.0.0"
)

# ── Response Schema ──────────────────────────────────────────────────────────

class ContactDetails(BaseModel):
    model_used: Optional[str] = None
    full_name: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    email: Optional[list[str]] = None
    phone: Optional[list[str]] = None
    website: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    twitter: Optional[str] = None
    other_social: Optional[dict] = None
    extra_info: Optional[str] = None


# ── Helper ────────────────────────────────────────────────────────────────────

async def extract_card_details(image_bytes: bytes, media_type: str) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set in environment")

    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = """You are an expert at reading visiting/business cards.

First, determine if this image is a visiting/business card.
If it is NOT a business card, return ONLY this JSON:
{"error": "Not a business card"}

If it IS a business card, extract ALL visible information and return ONLY this JSON:
{
  "full_name": "string",
  "job_title": "string",
  "company": "string",
  "email": ["list of emails"],
  "phone": ["list of phone numbers"],
  "website": "string",
  "address": "string",
  "linkedin": "string",
  "twitter": "string",
  "other_social": {"platform": "handle"},
  "extra_info": "any other text on the card"
}
Return ONLY the JSON. No explanation, no markdown fences."""

    payload = {
        "model": "google/gemma-3-4b-it:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://visiting-card-parser.app",
                "X-Title": "Visiting Card Parser"
            },
            json=payload
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter API error {response.status_code}: {response.text}"
        )

    result = response.json()
    raw_text = result["choices"][0]["message"]["content"].strip()
    print("RAW MODEL OUTPUT:", raw_text)
    print(f"Model used: {result.get('model', 'unknown')}")  # ← ADD THIS

    # Strip markdown fences if model adds them

    # Strip markdown fences if model adds them
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
        # Check if model says it's not a business card
        if "error" in parsed:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is not a visiting card. Please upload a valid business card image."
            )
        parsed["model_used"] = result.get("model", "unknown")
        return parsed
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Could not parse model response as JSON: {raw_text}"
        )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Visiting Card Parser API is running"}


@app.post(
    "/parse-card",
    response_model=ContactDetails,
    tags=["Card Parser"],
    summary="Upload a visiting card image to extract contact details"
)
async def parse_card(file: UploadFile = File(...)):
    """
    Upload a visiting/business card image (JPEG, PNG, WEBP, GIF).
    Returns structured JSON with all contact details found on the card.
    """
    allowed_types = {
        "image/jpeg": "image/jpeg",
        "image/jpg":  "image/jpeg",
        "image/png":  "image/png",
        "image/webp": "image/webp",
        "image/gif":  "image/gif"
    }

    content_type = file.content_type or ""
    media_type = allowed_types.get(content_type.lower())

    if not media_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Allowed: JPEG, PNG, WEBP, GIF"
        )

    image_bytes = await file.read()

    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size is 5 MB.")

    details = await extract_card_details(image_bytes, media_type)
    return JSONResponse(content=details)