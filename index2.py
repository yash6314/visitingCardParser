import os
import json
import base64
import httpx
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Load env
load_dotenv()

app = FastAPI(
    title="Visiting Card Parser API",
    version="1.0.0"
)

MODEL_NAME = "gemini-2.5-flash"


# ── Response Schema ─────────────────────────────────────────────

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


# ── Helper Function ─────────────────────────────────────────────

async def extract_card_details(image_bytes: bytes, media_type: str) -> dict:
    try:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

        # Convert to base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        url = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_NAME}:generateContent?key={api_key}"

        prompt = """
You are an expert at reading visiting/business cards.

Step 1:
Check if this image is a BUSINESS CARD.

If NOT a business card, return:
{
  "error": "Not a business card"
}

Step 2:
If it IS a business card, extract ALL details.

Rules:
- Return ONLY valid JSON
- No markdown
- No explanation

Format:
{
  "full_name": "string",
  "job_title": "string",
  "company": "string",
  "email": ["list"],
  "phone": ["list"],
  "website": "string",
  "address": "string",
  "linkedin": "string",
  "twitter": "string",
  "other_social": {"platform": "handle"},
  "extra_info": "string"
}
"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": media_type,
                                "data": b64_image
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, json=payload)

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini API error: {response.text}"
            )

        result = response.json()

        raw_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()

        print("RAW MODEL OUTPUT:", raw_text)

        # ── Try parsing ──
        try:
            parsed = json.loads(raw_text)
        except:
            # cleanup
            if "```" in raw_text:
                parts = raw_text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") or part.startswith("json"):
                        raw_text = part
                        break

                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()

            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                raw_text = match.group(0)

            parsed = json.loads(raw_text)

        # Check if NOT business card
        if "error" in parsed:
            
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is not a business card"
            )

        parsed["model_used"] = MODEL_NAME
        return parsed

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


# ── Routes ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Visiting Card Parser Running"}


@app.post("/parse-card", response_model=ContactDetails)
async def parse_card(file: UploadFile = File(...)):

    allowed_types = {
        "image/jpeg": "image/jpeg",
        "image/jpg": "image/jpeg",
        "image/png": "image/png",
        "image/webp": "image/webp"
    }

    content_type = file.content_type or ""
    media_type = allowed_types.get(content_type.lower())

    if not media_type:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )

    image_bytes = await file.read()

    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 5MB)"
        )

    details = await extract_card_details(image_bytes, media_type)
    return JSONResponse(content=details)