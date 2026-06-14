# ─── FormPilot FastAPI /fill endpoint ────────────────────────────────────────
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
import requests
import json
import os

app = FastAPI(title="FormPilot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── RAW REQUEST LOGGER — catches request BEFORE Pydantic validation ──────────
# This logs even 422 errors so we can see exactly what was sent
@app.middleware("http")
async def log_raw_requests(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/fill":
        body = await request.body()
        print(f"[RAW REQUEST BODY] {body.decode('utf-8')}")
    response = await call_next(request)
    return response

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

class Field(BaseModel):
    id: str
    label: str
    type: str
    options: List[str] = []
    tag: str = "INPUT"

class FillRequest(BaseModel):
    session_id: str = "default"   # optional now — defaults to "default"
    fields: List[Field]
    user_data: Dict[str, Any]
    entry_num: Optional[int] = None

class MappedField(BaseModel):
    id: str
    value: str
    type: str
    action: str

class FillResponse(BaseModel):
    session_id: str
    mapping: List[MappedField]
    fields_received: int
    fields_mapped: int

def call_groq(fields: List[Dict], user_data: Dict, entry_num: Optional[int] = None) -> List[Dict]:

    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment")

    fields_for_llm = [
        {
            "id":      f["id"],
            "label":   f["label"][:120],
            "type":    f["type"],
            "options": [o[:60] for o in f.get("options", [])][:15]
        }
        for f in fields
        if f.get("id") and (f.get("label", "").strip() or f.get("options"))
    ]

    if not fields_for_llm:
        return []

    entry_hint = ""
    if entry_num is not None:
        entry_hint = f"\nIMPORTANT: You are filling entry number {entry_num}. Use exp{entry_num} or edu{entry_num} data.\n"

    prompt = f"""You are a form filling agent. Map user data to form fields.
{entry_hint}
USER DATA:
{json.dumps(user_data, indent=2)}

FORM FIELDS:
{json.dumps(fields_for_llm, indent=2)}

Rules:
- Map every field that relates to user data — use semantic understanding
- For select fields, value MUST exactly match one of the available options
- Field ID is more reliable than label for mapping
- If user_data has a full name and form has separate first/last fields, split intelligently
- Only skip a field if there is genuinely no related data
- Return ONLY a JSON array. No explanation. No markdown. No code fences.

[{{"id":"field_id","value":"fill_value","type":"field_type","action":"type_or_select_or_check_or_radio"}}]"""

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type":  "application/json"
            },
            json={
                "model":       GROQ_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0
            },
            timeout=30
        )

        resp_json = response.json()

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq error: {resp_json}")

        raw = resp_json["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        start = raw.find("[")
        end   = raw.rfind("]")
        if start == -1 or end == -1:
            return []

        return json.loads(raw[start:end+1])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq call failed: {str(e)}")

@app.get("/")
def root():
    return {"status": "FormPilot API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug")
def debug():
    key = os.environ.get("GROQ_API_KEY", "NOT SET")
    return {
        "key_set": bool(key and key != "NOT SET"),
        "key_length": len(key),
        "key_start": key[:10] if key else "empty"
    }

@app.post("/fill", response_model=FillResponse)
def fill(req: FillRequest):
    if not req.fields:
        raise HTTPException(status_code=400, detail="No fields provided")
    if not req.user_data:
        raise HTTPException(status_code=400, detail="No user_data provided")

    print(f"[SESSION {req.session_id}] Fields received: {len(req.fields)}")
    print(f"[SESSION {req.session_id}] User data keys: {list(req.user_data.keys())}")
    print(f"[SESSION {req.session_id}] Full request: {json.dumps(req.model_dump(), indent=2)}")

    fields_dict = [f.model_dump() for f in req.fields]
    raw_mapping = call_groq(fields_dict, req.user_data, req.entry_num)

    print(f"[SESSION {req.session_id}] Groq mapping ({len(raw_mapping)} fields): {json.dumps(raw_mapping, indent=2)}")

    mapping = []
    for m in raw_mapping:
        if not m.get("id") or not m.get("value"):
            continue
        field_type = m.get("type", "text")
        action = m.get("action") or _infer_action(field_type)
        mapping.append(MappedField(
            id     = m["id"],
            value  = str(m["value"]),
            type   = field_type,
            action = action
        ))

    return FillResponse(
        session_id      = req.session_id,
        mapping         = mapping,
        fields_received = len(req.fields),
        fields_mapped   = len(mapping)
    )

def _infer_action(field_type: str) -> str:
    if field_type in ["select", "select-one"]:
        return "select"
    if field_type == "checkbox":
        return "check"
    if field_type == "radio":
        return "radio"
    return "type"
