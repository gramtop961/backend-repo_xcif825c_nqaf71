import os
import base64
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    query: str
    system_instruction: Optional[str] = None


def _gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server")
    return key


def _gemini_generate(contents: list) -> str:
    """Call Gemini Generative Language API via REST and return the text response."""
    api_key = _gemini_api_key()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent?key=" + api_key
    )
    payload = {"contents": contents}
    try:
        resp = requests.post(url, json=payload, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error contacting Gemini: {e}")
    if resp.status_code != 200:
        detail = resp.text[:500]
        raise HTTPException(status_code=resp.status_code, detail=f"Gemini API error: {detail}")
    data = resp.json()
    try:
        candidates = data.get("candidates") or []
        first = candidates[0]
        parts = first.get("content", {}).get("parts", [])
        # Concatenate all text parts if multiple
        texts = [p.get("text", "") for p in parts if "text" in p]
        text = "\n".join([t for t in texts if t])
        if not text:
            # Some responses may nest differently
            text = data.get("text") or ""
        if not text:
            text = "No response text returned from model."
        return text
    except Exception:
        return "Unable to parse model response. Raw: " + str(data)[:500]


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


@app.post("/api/solve/text")
def solve_text(body: TextRequest):
    user_instruction = body.system_instruction or (
        "You are a helpful math tutor. Solve the problem step-by-step. "
        "Show clear reasoning, intermediate steps, and final answer."
    )
    contents = [
        {"role": "user", "parts": [
            {"text": user_instruction},
            {"text": "Problem:"},
            {"text": body.query},
        ]}
    ]
    answer = _gemini_generate(contents)
    return {"answer": answer}


@app.post("/api/solve/image")
async def solve_image(
    image: UploadFile = File(...),
    query: Optional[str] = Form(None),
):
    # Read file bytes and base64 encode
    try:
        data = await image.read()
    finally:
        await image.close()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    mime = image.content_type or "image/png"
    b64 = base64.b64encode(data).decode("utf-8")
    instruction = (
        "You are a helpful math tutor. Read the problem from the image and solve it "
        "step-by-step. If the image is unclear, state assumptions."
    )
    parts = [
        {"text": instruction},
        {"inline_data": {"mime_type": mime, "data": b64}},
    ]
    if query:
        parts.append({"text": "Additional context:"})
        parts.append({"text": query})
    contents = [{"role": "user", "parts": parts}]
    answer = _gemini_generate(contents)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
