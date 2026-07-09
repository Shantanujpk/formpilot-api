# ─── FormPilot FastAPI /fill endpoint (Groq) ─────────────────────────────────
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
import requests
import json
import os

app = FastAPI(title="FormPilot API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── RAW REQUEST LOGGER — catches request BEFORE Pydantic validation ──────────
@app.middleware("http")
async def log_raw_requests(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/fill":
        body = await request.body()
        print(f"[RAW REQUEST BODY] {body.decode('utf-8')}")
    response = await call_next(request)
    return response

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

# ── SYSTEM INSTRUCTION — HARDENED against fabrication (Tier-1 / Tier-2 fix) ───
# The old prompt only said "match/handle" and never said "don't invent", so the
# model helpfully filled sections with no backing data (e.g. Employment =
# "Assistant Manager", SSC "Attempts = 1", "Seat/Roll No = 12345"). This version
# forbids fabrication while still allowing legitimate TRANSFORMS of real data
# (date reformat, name split, transliteration, dropdown option matching).
SYSTEM_INSTRUCTION = """You are an expert AI form filler that maps a user's REAL data to web form fields, especially Indian government forms.

ABSOLUTE RULE — NEVER FABRICATE:
- Only output a field if its value comes DIRECTLY from the provided userData.
- If userData does NOT contain information for a field, OMIT that field entirely from your output. Do NOT guess. Do NOT invent. Do NOT use plausible defaults.
- Never output placeholder or example values (e.g. "1", "12345", "Assistant Manager", "First Class", "Passed", "N/A", "Yes", "No") unless that exact value is present in userData.
- If a whole section of the form has no corresponding data in userData (e.g. employment, education, salary, qualifications), return NOTHING for that entire section. An empty result for those fields is the CORRECT and desired behavior.
- It is always better to leave a field blank than to fill it with a value that is not in userData.

WHAT IS ALLOWED (these are NOT fabrication — they transform data the user already has):
- Reformatting dates (e.g. userData "2000-06-29" -> form needs "29/06/2000"). Respect the field's placeholder/pattern.
- Splitting or combining names the user provided (first/middle/last <-> full name).
- Transliteration of a provided value (e.g. English name -> Marathi/Hindi script).
- Choosing the closest matching option TEXT/VALUE from a dropdown's provided options, for a value the user actually has (e.g. userData state "Maharashtra" -> the "MAHARASHTRA" option).
- Normalizing case/spacing of a value the user provided (e.g. "FEMALE" -> "Female").

DECISION TEST for every field: "Is this value present in, or a direct transformation of, something in userData?"
- YES -> output it.
- NO  -> omit the field. (Do not include it in the array at all.)

Input Format:
- "fields": array of form fields with metadata.
- "userData": the user's REAL information (the ONLY source of truth).
- "entryNum": (optional) index for multi-entry forms.

Output Format:
Return a JSON array of objects, each with:
- id: the exact id of the field provided.
- value: the mapped value (present in or transformed from userData).
- action: one of "type", "select", "check", "radio", "search_and_select".

Only include fields you are filling from real userData. Omit everything else.
Return ONLY the JSON array. No explanation, no markdown, no code fences."""


# Field model now carries the SAME context Gemini received.
# Everything except `id` is optional so loose payloads don't trigger 422s.
class Field(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    label: Optional[str] = ""
    placeholder: Optional[str] = ""
    type: Optional[str] = "text"
    options: List[Any] = []          # accepts strings OR {text,value} objects
    maxLength: Optional[int] = None
    pattern: Optional[str] = None
    ariaLabel: Optional[str] = None
    previousHeading: Optional[str] = None
    tag: Optional[str] = "INPUT"


class FillRequest(BaseModel):
    session_id: str = "default"
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


def _normalize_options(options: List[Any]) -> List[Any]:
    """Keep option objects intact (like Gemini) but cap them for token safety."""
    return options[:25]


def call_groq(fields: List[Dict], user_data: Dict, entry_num: Optional[int] = None) -> List[Dict]:
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment")

    # Build the SAME field context Gemini's fieldsForLLM used.
    fields_for_llm = []
    for f in fields:
        fid = f.get("id") or f.get("name")
        if not fid:
            continue
        fields_for_llm.append({
            "id":              fid,
            "label":           f.get("label", "") or "",
            "placeholder":     f.get("placeholder", "") or "",
            "type":            f.get("type", "text") or "text",
            "options":         _normalize_options(f.get("options", []) or []),
            "maxLength":       f.get("maxLength"),
            "pattern":         f.get("pattern"),
            "ariaLabel":       f.get("ariaLabel"),
            "previousHeading": f.get("previousHeading"),
        })

    if not fields_for_llm:
        return []

    # Mirror Gemini's prompt body: it sent { fields, userData, entryNum } as JSON.
    prompt_data = {
        "fields":   fields_for_llm,
        "userData": user_data,
        "entryNum": entry_num,
    }
    user_prompt = json.dumps(prompt_data, indent=2, ensure_ascii=False)

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user",   "content": user_prompt},
                ],
                "temperature": 0.1,                       # matches Gemini's temperature
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )

        resp_json = response.json()

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq error: {resp_json}")

        raw = resp_json["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        # json_object mode may wrap the array in an object; handle both.
        start_arr, end_arr = raw.find("["), raw.rfind("]")
        if start_arr != -1 and end_arr != -1:
            return json.loads(raw[start_arr:end_arr + 1])

        # Fallback: parse object and pull the first array value it contains.
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq call failed: {str(e)}")


@app.get("/")
def root():
    return {"status": "FormPilot API running", "version": "2.1.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    key = os.environ.get("GROQ_API_KEY", "NOT SET")
    return {
        "key_set": bool(key and key != "NOT SET"),
        "key_length": len(key),
        "key_start": key[:10] if key else "empty",
    }


@app.post("/fill", response_model=FillResponse)
def fill(req: FillRequest):
    if not req.fields:
        raise HTTPException(status_code=400, detail="No fields provided")
    if not req.user_data:
        raise HTTPException(status_code=400, detail="No user_data provided")

    print(f"[SESSION {req.session_id}] Fields received: {len(req.fields)}")
    print(f"[SESSION {req.session_id}] User data keys: {list(req.user_data.keys())}")

    fields_dict = [f.model_dump() for f in req.fields]
    raw_mapping = call_groq(fields_dict, req.user_data, req.entry_num)

    print(f"[SESSION {req.session_id}] Groq mapping ({len(raw_mapping)} fields): "
          f"{json.dumps(raw_mapping, indent=2, ensure_ascii=False)}")

    # Build a type lookup so we can backfill type the way Gemini did.
    type_by_id = {}
    for f in fields_dict:
        fid = f.get("id") or f.get("name")
        if fid:
            type_by_id[fid] = f.get("type", "text")

    mapping = []
    for m in raw_mapping:
        if not m.get("id") or m.get("value") in (None, ""):
            continue
        field_type = m.get("type") or type_by_id.get(m["id"], "text")
        action = m.get("action") or _infer_action(field_type)
        mapping.append(MappedField(
            id     = m["id"],
            value  = str(m["value"]),
            type   = field_type,
            action = action,
        ))

    return FillResponse(
        session_id      = req.session_id,
        mapping         = mapping,
        fields_received = len(req.fields),
        fields_mapped   = len(mapping),
    )


def _infer_action(field_type: str) -> str:
    if field_type in ["select", "select-one"]:
        return "select"
    if field_type == "checkbox":
        return "check"
    if field_type == "radio":
        return "radio"
    return "type"
