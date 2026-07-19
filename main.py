# ─── FormPilot FastAPI /fill endpoint (Cerebras) ─────────────────────────────
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
# PROVIDER: Cerebras (US-based, OpenAI-compatible). Chosen over Groq because
# Groq's free tier TPM (12K/min, 100K/day) kept rate-limiting us and the paid
# Developer upgrade was unavailable. Cerebras free tier gives ~1M tokens/hour
# AND ~1M/day, 30K TPM, 65,536 context on gpt-oss-120b — far more headroom.
#
# MODEL: gpt-oss-120b (OpenAI's open 120B model, "Production" on Cerebras).
#
# ENV VAR NEEDED (Render / Railway):  CEREBRAS_API_KEY
#
# ── THIS REVISION (3.2.0): DETERMINISTIC DATE NORMALISATION ──────────────────
# Extraction testing showed date-format drift across five documents:
#     "30 SEP 2014"  "21-June-2005"  "16-01-2017"  "14.06.2016"
#     "16th day of April in the year 1998"
# Root cause: the extraction prompts said "return exactly as printed, in
# DD/MM/YYYY format" — a CONTRADICTION when the printed date isn't numeric, and
# the model resolved it as "as printed". Asking a model to reformat is the wrong
# tool: it works ~90% of the time, which is the hardest failure mode to catch.
#
# FIX: date conversion is a pure mechanical transform, so it belongs in CODE.
# _normalize_dates() runs over user_data at ingest and rewrites every date-ish
# value to canonical DD/MM/YYYY before the mapper ever sees it. The extraction
# prompt can now simply say "return the date as printed" — the model READS, the
# code CONVERTS. Same hybrid principle that fixed the cascade bug: deterministic
# work in code, judgement in the LLM.
#
# NOTE: this is LAYER 1 (canonical storage). LAYER 2 is field-aware formatting at
# fill time in content.js — reshaping DD/MM/YYYY into whatever the target field
# wants (text DD/MM/YYYY, ISO YYYY-MM-DD for <input type="date">, or split across
# day/month/year dropdowns). A single canonical form makes layer 2 trivial.

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
import requests
import json
import os
import re

app = FastAPI(title="FormPilot API", version="3.2.0")

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
LLM_MODEL = "gpt-oss-120b"
# LLM_MODEL = "gemma-4-31b"          # <- smaller free fallback

LLM_URL = "https://api.cerebras.ai/v1/chat/completions"
LLM_API_KEY_ENV = "CEREBRAS_API_KEY"

# Cerebras gpt-oss-120b returns a JSON-SCHEMA fragment when response_format is
# set, instead of our data. Disabled; we rely on the prompt.
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
- DROPDOWN ADMIN-LEVEL MATCHING (applies even when the dropdown arrives alone): when a select field represents an administrative level (district, taluka, tehsil, sub-district, division, block) and ANY value in userData (especially "city", "state", "locality", or a segment of "fulladdress") EXACTLY matches one of the field's provided options, select that option. Indian district dropdowns commonly list the city name as a district (e.g. userData city "Pune" -> district option "Pune"; "Nagpur", "Mumbai" likewise). Do not skip the field just because there is no userData key literally named "district" — match by option text against the place values you have. Only omit if none of the field's options matches any place value in userData.

════════ DATES ════════
Every date in userData has ALREADY been normalised to DD/MM/YYYY (day first, then month, then year) before it reached you. Trust that order — the first number is the DAY, the second is the MONTH. Never re-interpret a date as month-first.
Reshape a date only to match what the target FIELD requires:
- a text field showing DD/MM/YYYY or with no stated format -> pass it through unchanged
- a field whose placeholder/pattern states another order (e.g. YYYY-MM-DD, DD-MM-YYYY) -> rearrange to match, keeping the same day, month and year
- separate day / month / year fields or dropdowns -> split the value and put each part in its own field
Never invent a missing part of a date, and never change which day or month the value refers to.

════════ CERTIFIED STATUS vs ORDINARY FACT — NEVER INFER A STATUS ════════
Distinguish two kinds of field:
- A FACT is something printed on a document: a name, an address, a date, a place, a number, a board, a caste name. If userData contains it, use it.
- A STATUS is an official standing, entitlement, membership, eligibility, category, or exemption that only a competent authority can confer, and that a specific document EXISTS to certify (domicile/residency status, reservation or quota category, disability status, employment with a named organisation, ex-serviceman standing, minority status, income/EWS eligibility, sports or cultural quota, freedom-fighter descent, and any similar claim).

A STATUS field may ONLY be answered from a document that actually CERTIFIES that status and is present in userData. If that certifying document was not uploaded, OMIT the field. There is no other legitimate source.

NEVER infer a status from a related fact. Facts and statuses are not interchangeable, and a correlation is not evidence. Specifically, and by the same logic for every equivalent case:
- An address, city, district, state, pincode, or place of birth is NOT evidence of domicile, residency status, local/regional quota eligibility, or nationality. Living somewhere, being born somewhere, or studying somewhere is not the same as being certified as domiciled or resident there.
- A caste name is NOT evidence of a reservation category, and a category is NOT evidence of a caste. Use only what a caste/validity certificate states.
- A religion, surname, language, or gender is NOT evidence of minority status, category, or any entitlement.
- An employer, school, board, or institution named on a document is NOT evidence of being an employee of any particular government body or organisation named in the form.
- A date of birth is NOT evidence of eligibility for an age relaxation; marks or income are NOT evidence of a quota or waiver.
- The absence of evidence is NOT evidence of absence: do not answer "No" to a status question just because userData contains nothing supporting "Yes". Both "Yes" and "No" are claims. Omit instead.

If a status question offers Yes/No, a category list, or any set of options, and no uploaded document certifies the answer, return NOTHING for that field. A blank status field is CORRECT and expected — the user will answer it themselves. A guessed status is a false declaration on a government form and is the most serious failure this system can produce.

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

════════ THE "hint" FIELD — READ IT AND OBEY IT ════════
A field may carry a "hint": the page's own helper/instruction text for that field (plus "label" and "previousHeading" for context). The hint is an INSTRUCTION FROM THE FORM ITSELF and outranks your default choice of value.

- A hint often states WHICH DOCUMENT the value must come from — e.g. "enter your name as per 10th marksheet", "as printed on your PAN card", "as per Aadhaar". When it does, use the value that came from THAT document, even if userData holds a different value of the same kind (a person's name on their Aadhaar and on their marksheet routinely differ in order or spelling — they are not interchangeable).
- A hint may also state a FORMAT or CONSTRAINT ("in capital letters", "DD/MM/YYYY", "as per SSC certificate"). Honour it.
- If the hint demands a source or form of the value that userData does NOT contain, OMIT the field. Do NOT substitute a value from a different document or a differently-shaped value — a blank field is correct; a value from the wrong source is a critical failure.
- If a field has no hint, fall back to the normal rules above.

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
    hint: Optional[str] = ""         # page helper text, e.g. "name as per 10th marksheet"
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


# ═════════════════════════════════════════════════════════════════════════════
#  DATE NORMALISATION  (deterministic — no LLM involved)
# ═════════════════════════════════════════════════════════════════════════════

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# A key is treated as a date if it mentions one of these...
_DATE_KEY_HINTS = (
    "date", "dob", "birthday", "issued", "admission",
    "leaving", "validity", "expiry", "registration",
)

# ...unless it also mentions one of these (identifiers, places, names, and the
# standalone day/month/year part-fields, which must NOT be rewritten).
_DATE_KEY_EXCLUDE = (
    "number", "_no", "_id", "authority", "place",
    "district", "state", "taluka", "name", "status", "grade",
)


def _is_date_key(key: str) -> bool:
    """Whitelist of keys whose values should be date-normalised."""
    if not isinstance(key, str):
        return False
    k = key.lower()
    # Standalone components (birthday_day, passing_year, ...) stay untouched.
    if k.endswith("_year") or k.endswith("_month") or k.endswith("_day"):
        return False
    if any(x in k for x in _DATE_KEY_EXCLUDE):
        return False
    return any(h in k for h in _DATE_KEY_HINTS)


def _four_digit_year(y: int) -> int:
    if y >= 1000:
        return y
    if y < 100:
        return 2000 + y if y <= 30 else 1900 + y   # 2-digit year window
    return y


def normalize_date(value: Any) -> Optional[str]:
    """
    Convert a printed date into canonical DD/MM/YYYY.
    Returns None when the value is not recognisably a date, so the caller can
    keep the original untouched. NEVER fabricates a missing part.

    Indian documents are DAY-FIRST. Ambiguous all-numeric input therefore stays
    day-first; only an impossible day (>12 in the month slot) triggers a swap.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    low = s.lower()

    # 1. Worded: "16th day of April in the year 1998"
    m = re.search(
        r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:day\s+of\s+)?"
        r"([A-Za-z]{3,9})\s*(?:,)?\s*(?:in\s+the\s+year\s+)?(\d{2,4})",
        low,
    )
    if m and m.group(2) in _MONTHS:
        d, mon, y = int(m.group(1)), _MONTHS[m.group(2)], _four_digit_year(int(m.group(3)))
        if 1 <= d <= 31:
            return f"{d:02d}/{mon:02d}/{y:04d}"

    # 2. Day + month-name: "30 SEP 2014", "21-June-2005", "05 June, 2023"
    m = re.search(r"(\d{1,2})\s*[-/.\s]\s*([A-Za-z]{3,9})\s*[-/.,\s]\s*(\d{2,4})", low)
    if m and m.group(2) in _MONTHS:
        d, mon, y = int(m.group(1)), _MONTHS[m.group(2)], _four_digit_year(int(m.group(3)))
        if 1 <= d <= 31:
            return f"{d:02d}/{mon:02d}/{y:04d}"

    # 3. Month-name first: "April 16, 1998", "Sep 30 2014"
    m = re.search(
        r"([A-Za-z]{3,9})\s*[-/.\s]\s*(\d{1,2})\s*(?:st|nd|rd|th)?\s*[-/.,\s]\s*(\d{2,4})",
        low,
    )
    if m and m.group(1) in _MONTHS:
        mon, d, y = _MONTHS[m.group(1)], int(m.group(2)), _four_digit_year(int(m.group(3)))
        if 1 <= d <= 31:
            return f"{d:02d}/{mon:02d}/{y:04d}"

    # 4. All-numeric: 16-01-2017, 14.06.2016, 09/09/2014, 5/6/2023, 16/01/17, 2014-09-09
    m = re.search(r"(\d{1,4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,4})", s)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if len(m.group(1)) == 4:                 # ISO: YYYY-MM-DD
            y, mon, d = a, b, c
        else:
            d, mon, y = a, b, c
            if d > 12 and mon > 12:
                return None                      # nonsense, leave the original alone
            if d <= 12 and mon > 12:
                d, mon = mon, d                  # clearly month-first -> swap to day-first
            y = _four_digit_year(y)
        if 1 <= d <= 31 and 1 <= mon <= 12 and 1000 <= y <= 2999:
            return f"{d:02d}/{mon:02d}/{y:04d}"

    return None


def _normalize_dates(data: Dict[str, Any], session_id: str = "") -> Dict[str, Any]:
    """
    Rewrite every date-ish value in userData to canonical DD/MM/YYYY.
    Fails safe: a value that cannot be parsed is left exactly as it arrived.
    """
    if not isinstance(data, dict):
        return data
    out: Dict[str, Any] = {}
    for k, v in data.items():
        if _is_date_key(k) and isinstance(v, (str, int, float)):
            fixed = normalize_date(v)
            if fixed and fixed != str(v).strip():
                print(f"[SESSION {session_id}] DATE NORMALISED {k}: {v!r} -> {fixed!r}")
                out[k] = fixed
                continue
        out[k] = v
    return out


# ═════════════════════════════════════════════════════════════════════════════


def _normalize_options(options: List[Any]) -> List[Any]:
    """Keep option objects intact but cap them for token safety."""
    return options[:25]


def _clean_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    KEY HYGIENE: extraction sometimes emits malformed keys — trailing spaces
    ("nationality "), mixed case and spaces ("Year of Passing"). The LLM tolerates
    these, but any deterministic matcher (e.g. localFillSelects in content.js) would
    miss them. Normalise to trimmed, lowercase, space-free keys. First key wins on
    a collision, so nothing is silently overwritten.
    """
    if not isinstance(data, dict):
        return data
    cleaned: Dict[str, Any] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            cleaned[k] = v
            continue
        nk = k.strip().lower().replace(" ", "")
        if nk and nk not in cleaned:
            cleaned[nk] = v
        elif nk not in cleaned:
            cleaned[k] = v
    return cleaned


def _lean_user_data(user_data: Dict[str, Any], session_id: str = "") -> Dict[str, Any]:
    """
    PAYLOAD TRIM: the mapper only needs the user's actual values.
    Drops baggage the model never uses (formFillInstructions = stale old-path
    output, processedDocuments = raw per-doc data already merged, formId = routing
    id). Sends ONLY extractedFields (+ manual top-level values). Falls back to the
    whole user_data if extractedFields isn't present, so nothing breaks whether the
    payload is fat (old clients) or lean (trimmed popup.js).

    Then applies key hygiene and DATE NORMALISATION, so the mapper always sees
    clean keys and canonical DD/MM/YYYY dates.
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
        return _normalize_dates(_clean_keys(lean), session_id)

    return _normalize_dates(
        _clean_keys({
            k: v for k, v in user_data.items()
            if k not in ("formFillInstructions", "processedDocuments")
        }),
        session_id,
    )


def _extract_mapping_json(raw: str) -> List[Dict]:
    """
    Robustly pull the mapping array out of a model response that may contain
    trailing text, leading reasoning, or an object wrapper. Uses raw_decode so
    'valid JSON + extra data' no longer throws.
    """
    if not raw:
        return []

    decoder = json.JSONDecoder()

    for i, ch in enumerate(raw):
        if ch not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(raw[i:])
        except ValueError:
            continue

        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, list):
                    return v
            if "id" in value:
                return [value]

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except ValueError:
        pass

    print("[PARSE WARNING] could not extract a JSON array from model output")
    return []


def call_llm(fields: List[Dict], user_data: Dict, entry_num: Optional[int] = None,
             session_id: str = "") -> List[Dict]:
    api_key = os.environ.get(LLM_API_KEY_ENV, "")
    if not api_key:
        raise HTTPException(status_code=500, detail=f"{LLM_API_KEY_ENV} not set in environment")

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
            "hint":            f.get("hint", "") or "",
        })

    if not fields_for_llm:
        return []

    lean_user_data = _lean_user_data(user_data, session_id)

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

        print(f"[LLM RAW STATUS] {response.status_code}")
        print(f"[LLM RAW RESPONSE] {json.dumps(resp_json, ensure_ascii=False)[:4000]}")

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"LLM error: {resp_json}")

        raw = resp_json["choices"][0]["message"]["content"].strip()
        print(f"[LLM MESSAGE CONTENT] {raw[:2000]}")
        raw = raw.replace("```json", "").replace("```", "").strip()

        return _extract_mapping_json(raw)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")


@app.get("/")
def root():
    return {"status": "FormPilot API running", "version": "3.2.0",
            "provider": "cerebras", "model": LLM_MODEL,
            "hint_support": True, "date_normalisation": True}


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


@app.post("/debug/date")
def debug_date(payload: Dict[str, Any]):
    """
    Quick sanity endpoint for the date normaliser.
    POST {"value": "30 SEP 2014"} -> {"input": ..., "normalised": "30/09/2014"}
    """
    v = payload.get("value")
    return {"input": v, "normalised": normalize_date(v)}


@app.post("/fill", response_model=FillResponse)
def fill(req: FillRequest):
    if not req.fields:
        raise HTTPException(status_code=400, detail="No fields provided")
    if not req.user_data:
        raise HTTPException(status_code=400, detail="No user_data provided")

    print(f"[SESSION {req.session_id}] Fields received: {len(req.fields)}")
    print(f"[SESSION {req.session_id}] User data keys: {list(req.user_data.keys())}")

    fields_dict = [f.model_dump() for f in req.fields]

    # QA VISIBILITY: confirm the scanner's context is actually arriving. If
    # labelled/hinted are 0, the scanner fix isn't loaded in the browser.
    labelled = sum(1 for f in fields_dict if f.get("label"))
    hinted = sum(1 for f in fields_dict if f.get("hint"))
    print(f"[SESSION {req.session_id}] Context: {labelled}/{len(fields_dict)} labelled, {hinted} hinted")
    for f in fields_dict:
        if f.get("hint"):
            print(f"[SESSION {req.session_id}]   HINT {f.get('id')}: {f.get('hint')}")

    raw_mapping = call_llm(fields_dict, req.user_data, req.entry_num, req.session_id)

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
