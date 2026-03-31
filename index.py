import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Load env
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(
    title="Visiting Card Parser API",
    version="1.0.0"
)
modelName = "gemini-flash-latest"

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
        model = genai.GenerativeModel(modelName)

        prompt = """
You are an expert at reading visiting/business cards.
Extract ALL visible information and return STRICTLY valid JSON only.

Rules:
- Do NOT include ``` or markdown
- Do NOT add explanations
- Return ONLY JSON

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

        response = model.generate_content([
            {
                "mime_type": media_type,
                "data": image_bytes
            },
            prompt
        ])

        raw_text = response.text.strip()

        print("RAW MODEL OUTPUT:", raw_text)  # Debug log

        # ── Step 1: Try direct JSON parse ──
        try:
            parsed = json.loads(raw_text)
            parsed["model_used"] = modelName
            return parsed
        except:
            pass

        # ── Step 2: Remove markdown if exists ──
        if "```" in raw_text:
            parts = raw_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") or part.startswith("json"):
                    raw_text = part
                    break

            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        # ── Step 3: Extract JSON using regex ──
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            raw_text = match.group(0)

        # ── Step 4: Final parse ──
        parsed = json.loads(raw_text)
        parsed["model_used"] = modelName
        return parsed

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# ── Routes ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Gemini Visiting Card Parser running"}


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
            detail="Unsupported file type. Use JPEG, PNG, WEBP"
        )

    image_bytes = await file.read()

    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 5MB)"
        )

    details = await extract_card_details(image_bytes, media_type)
    return JSONResponse(content=details)