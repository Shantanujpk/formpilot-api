# ─── FormPilot FastAPI /fill endpoint ────────────────────────────────────────
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Test: POST http://localhost:8000/fill

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import requests
import json
import re
import os
app = FastAPI(title="FormPilot API", version="1.0.0")

# ─── CORS — allow Chrome extension to call this API ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # lock this down to your extension origin in prod
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─── GROQ CONFIG ─────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

# ─── REQUEST / RESPONSE MODELS ───────────────────────────────────────────────

class Field(BaseModel):
    id: str
    label: str
    type: str
    options: list[str] = []
    tag: str = "INPUT"

class FillRequest(BaseModel):
    session_id: str                  # any string — used for logging
    fields: list[Field]
    user_data: dict[str, Any]
    entry_num: int | None = None     # pass 2,3,4 for multi-entry sections

class MappedField(BaseModel):
    id: str
    value: str
    type: str
    action: str                      # "type" | "select" | "check" | "radio"

class FillResponse(BaseModel):
    session_id: str
    mapping: list[MappedField]
    fields_received: int
    fields_mapped: int

# ─── GROQ CALL ───────────────────────────────────────────────────────────────
def call_groq(fields: list[dict], user_data: dict, entry_num: int | None = None) -> list[dict]:

    # Trim fields for token efficiency
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

    prompt = f"""You are a form filling agent. Map user data to form fields semantically.
{entry_hint}
USER DATA:
{json.dumps(user_data, indent=2)}

FORM FIELDS:
{json.dumps(fields_for_llm, indent=2)}

Rules:
- Be aggressive — map every field that reasonably relates to user data
- email → email, phone → phone/mobile, city → city, address → address_line1
- state → state, country → country, postal code / pincode → postal_code
- first name → first_name, last name → last_name, middle name → middle_name
- father name → father_name, mother name → mother_name
- date of birth / dob → dob
- gender → gender, nationality → nationality
- linkedin → linkedin_url, website → website_url
- skills → use skills_list as comma string
- field of study / major → edu field value
- visa details / provide details → visa_details value ONLY
- ethnicity → ethnicity_us, disability → disability
- veteran → veteran_status
- same as / same as permanent → true (agent handles automatically)
- For select fields value MUST exactly match one of the available options
- work authorization → Yes, sponsorship → Yes, non-compete → No
- government employee → No, export control → No
- acknowledge or agree → Yes, pick matching option
- If two fields have identical labels use field_id to differentiate
- HeightInCMS → height value, Weight → weight value
- Field ID is more reliable than label for semantic mapping
- percentageGrade → percentage, expYears → years of experience
- payScale / gradePay / basicPay → respective salary fields
- Only skip if genuinely no related data exists
- Return ONLY a JSON array. No explanation. No markdown. No code fences.

[{{"id":"field_id","value":"fill_value","type":"field_type","action":"type_or_select_or_check_or_radio"}}]"""

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
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

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "FormPilot API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/fill", response_model=FillResponse)
def fill(req: FillRequest):
    """
    Main endpoint. Chrome extension sends scanned fields + user data.
    Returns field mapping with values and actions for the extension to execute.

    Request:
    {
        "session_id": "abc123",
        "fields": [{"id": "firstName", "label": "First Name", "type": "text", "options": [], "tag": "INPUT"}],
        "user_data": {"first_name": "Shantanu", ...},
        "entry_num": null
    }

    Response:
    {
        "session_id": "abc123",
        "mapping": [{"id": "firstName", "value": "Shantanu", "type": "text", "action": "type"}],
        "fields_received": 1,
        "fields_mapped": 1
    }
    """

    if not req.fields:
        raise HTTPException(status_code=400, detail="No fields provided")

    if not req.user_data:
        raise HTTPException(status_code=400, detail="No user_data provided")

    # Convert Pydantic models to dicts for Groq
    fields_dict = [f.model_dump() for f in req.fields]

    # Call Groq
    raw_mapping = call_groq(fields_dict, req.user_data, req.entry_num)

    # Normalise — ensure action field exists
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
        session_id     = req.session_id,
        mapping        = mapping,
        fields_received = len(req.fields),
        fields_mapped  = len(mapping)
    )

def _infer_action(field_type: str) -> str:
    if field_type in ["select", "select-one"]:
        return "select"
    if field_type == "checkbox":
        return "check"
    if field_type == "radio":
        return "radio"
    return "type"

