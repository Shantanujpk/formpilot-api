# ─── FormPilot FastAPI /fill endpoint (Cerebras) ─────────────────────────────
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
# PROVIDER: Cerebras (US-based, OpenAI-compatible). Chosen over Groq because
# Groq's free tier TPM (12K/min, 100K/day) kept rate-limiting us and the paid
# Developer upgrade was unavailable. Cerebras free tier gives ~1M tokens/hour
# AND ~1M/day, 30K TPM, 65,536 context on gpt-oss-120b — far more headroom.
#
# MODEL: gpt-oss-120b (OpenAI's open 120B model, "Production" on Cerebras).
# Bigger than the Llama 3.3 70B we ran on Groq, so it should follow the
# no-fabricate rule at least as well and fix the 8B hallucination.
#
# ENV VAR NEEDED ON RAILWAY:  CEREBRAS_API_KEY
# (You can leave the old GROQ_API_KEY set too — it's just unused now.)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
import requests
import json
import os

app = FastAPI(title="FormPilot API", version="3.0.0")

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

# ── PROVIDER / MODEL SELECTION (Cerebras, OpenAI-compatible) ─────────────────
# One-line toggle. gpt-oss-120b is the biggest + Production-grade free model.
# If its mapping style doesn't suit your forms, try "gemma-4-31b" (also free).
LLM_MODEL = "gpt-oss-120b"
# LLM_MODEL = "gemma-4-31b"          # <- smaller free fallback

LLM_URL = "https://api.cerebras.ai/v1/chat/completions"
# Cerebras uses the same key env var name we set on Railway:
LLM_API_KEY_ENV = "CEREBRAS_API_KEY"

# Some providers/models reject or misinterpret the response_format param.
# Cerebras gpt-oss-120b returns a JSON-SCHEMA fragment ({"type":"object"}) when
# response_format is set, instead of our data. So we DISABLE it and rely on the
# prompt (which already says "return ONLY a JSON array"). Cerebras also supports
# passing a strict schema via response_format, but prompt-driven JSON is simpler
# and works here.
USE_JSON_RESPONSE_FORMAT = False

# ── SYSTEM INSTRUCTION — HARDENED against fabrication (Tier-1 / Tier-2 fix) ───
SYSTEM_INSTRUCTION = """You are a form-filling engine for Indian government and enrollment web forms. You receive a list of form fields and a user's real data object (userData). You decide which value, if any, goes into each field. You work on ANY such form on ANY website — never assume a specific form, site, or field naming scheme. Reason from the field's meaning and the data you are given, not from memorized layouts.

════════ THE ONE RULE ════════
DERIVE, NEVER INVENT.

A value is DERIVED (allowed) when you can name the exact userData key(s) it comes from. You may apply logic, arithmetic, formatting, or general world knowledge to RESHAPE, LOCATE, CONVERT, or MATCH that data to the field.
A value is INVENTED (forbidden) when no userData key supports it — even if it is a typical, likely, or default answer.

For EVERY field, ask: "Which userData key(s) does this value trace back to?"
- You can name them  -> output the field and record them in "source".
- You cannot         -> OMIT the field entirely. A blank field is correct and expected. An invented value is a critical failure.
When in doubt, omit.

════════ WHAT COUNTS AS DERIVED (allowed transformations) ════════
Judge by MEANING, not by matching key names to field names. The userData key and the field label will often differ in wording, language, or granularity — that is normal.
- Format/convert a value to the field's required form (dates to the field's pattern, numbers to strings, units, case, spacing).
- Split or combine values (e.g. a full name into parts, or parts into a whole).
- Compute a value that is a pure function of provided data (e.g. an age from a date of birth).
- Transliterate a provided value into the script the field expects.
- Match a provided value to the closest option of a dropdown/radio (by option text or value).
- DECOMPOSE COMPOSITE FIELDS: when the user has a combined value (a full postal address, a full name, or any multi-part string) and the form asks for its individual components, extract each component from the combined value. For an address, split it on its separators (commas, dashes) and map each segment to the field it fits: house/door number, building/society, road/area, landmark, village/town/city, taluka/tehsil/sub-district, district, state, pincode. Work from the WHOLE combined value, not just one convenient token.
- ONE VALUE CAN FILL SEVERAL FIELDS: a single real-world entity can be the correct answer for more than one differently-named field. If the data provides only one place name and the form asks for several administrative levels (e.g. both District and Village/City, or both Taluka and District) with no finer value to separate them, use that same place name for each field it correctly answers rather than filling one and leaving the rest blank. Do not "use up" a value on a single field. This reuse is DERIVATION (the value traces to real data), not invention.
- Only omit an address-level field when the combined value genuinely contains nothing that fits it.

════════ WHAT COUNTS AS INVENTED (never output) ════════
- Any value for a field about a topic the userData does not cover (e.g. education, employment, family, financial, or eligibility details that were never provided).
- Any yes/no, status, or category answer to a question the data does not answer — supplying even a "safe" default like "No" is invention.
- Statistically likely guesses that are not stated in the data (do not infer one attribute from a correlated one).
- Placeholder or example values of any kind.
- If an entire section of the form has no backing in userData, return nothing for that whole section. Empty output for it is CORRECT behavior, not a failure.

════════ FIELD MECHANICS ════════
- Echo each field's id back exactly as provided.
- Respect maxLength and pattern constraints.
- For a dropdown/radio, the value MUST be one of that field's provided options; if no option reasonably matches the user's value, omit the field.
- Cascading fields: you may only receive a parent field now (e.g. a top-level region); its dependent children appear in LATER requests after the parent is set. Map only fields present in the CURRENT request; never pre-map a field you were not given.

════════ OUTPUT ════════
Return ONLY a raw JSON array — start with [ and end with ]. No reasoning, no explanation, no markdown, no code fences, no schema object.
Each element: {"id": "<exact field id>", "value": "<derived value>", "action": "type|select|check|radio|search_and_select", "source": "<comma-separated userData key(s) this value derives from>"}
Include only fields you can source from userData. Omit all others. If nothing can be sourced, return []."""


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
    """Keep option objects intact but cap them for token safety."""
    return options[:25]


def _lean_user_data(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    PAYLOAD TRIM: the mapper only needs the user's actual values.
    Drops baggage the model never uses (formFillInstructions = stale old-path
    output, processedDocuments = raw per-doc data already merged, formId = routing
    id). Sends ONLY extractedFields (+ manual top-level values). Falls back to the
    whole user_data if extractedFields isn't present, so nothing breaks whether the
    payload is fat (old clients) or lean (trimmed popup.js).
    """
    if not isinstance(user_data, dict):
        return user_data

    extracted = user_data.get("extractedFields")
    if isinstance(extracted, dict) and extracted:
        lean = dict(extracted)
        for k, v in user_data.items():
            if k in ("extractedFields", "formFillInstructions", "processedDocuments", "formId"):
                continue
            if k not in lean and isinstance(v, (str, int, float, bool)):
                lean[k] = v
        return lean

    return {
        k: v for k, v in user_data.items()
        if k not in ("formFillInstructions", "processedDocuments")
    }


def call_llm(fields: List[Dict], user_data: Dict, entry_num: Optional[int] = None) -> List[Dict]:
    api_key = os.environ.get(LLM_API_KEY_ENV, "")
    if not api_key:
        raise HTTPException(status_code=500, detail=f"{LLM_API_KEY_ENV} not set in environment")

    # Build the field context the model maps against.
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

    lean_user_data = _lean_user_data(user_data)

    prompt_data = {
        "fields":   fields_for_llm,
        "userData": lean_user_data,
        "entryNum": entry_num,
    }
    user_prompt = json.dumps(prompt_data, indent=2, ensure_ascii=False)

    try:
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        # Some models/providers reject response_format; make it toggleable.
        if USE_JSON_RESPONSE_FORMAT:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(
            LLM_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=60,
        )

        resp_json = response.json()

        # DEBUG: log the FULL raw provider response so we can see exactly what came back.
        print(f"[LLM RAW STATUS] {response.status_code}")
        print(f"[LLM RAW RESPONSE] {json.dumps(resp_json, ensure_ascii=False)[:4000]}")

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"LLM error: {resp_json}")

        raw = resp_json["choices"][0]["message"]["content"].strip()
        print(f"[LLM MESSAGE CONTENT] {raw[:2000]}")
        raw = raw.replace("```json", "").replace("```", "").strip()

        start_arr, end_arr = raw.find("["), raw.rfind("]")
        if start_arr != -1 and end_arr != -1:
            return json.loads(raw[start_arr:end_arr + 1])

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
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")


@app.get("/")
def root():
    return {"status": "FormPilot API running", "version": "3.0.0",
            "provider": "cerebras", "model": LLM_MODEL}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    key = os.environ.get(LLM_API_KEY_ENV, "NOT SET")
    return {
        "provider": "cerebras",
        "model": LLM_MODEL,
        "key_env": LLM_API_KEY_ENV,
        "key_set": bool(key and key != "NOT SET"),
        "key_length": len(key),
        "key_start": key[:8] if key else "empty",
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
    raw_mapping = call_llm(fields_dict, req.user_data, req.entry_num)

    print(f"[SESSION {req.session_id}] Mapping ({len(raw_mapping)} fields): "
          f"{json.dumps(raw_mapping, indent=2, ensure_ascii=False)}")

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
