# app.py — Analytics microservice for Moodle Adaptive Quiz
import os, json, time, traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# ───────────────────────────────────────────────────────────────
# ⚙️  Configuration
# ───────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "on").strip().lower() == "on"

app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=OPENAI_API_KEY)

# ───────────────────────────────────────────────────────────────
# 🩺 Health endpoint
# ───────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "model": MODEL})

# ───────────────────────────────────────────────────────────────
# 🔍 /analyze endpoint
# ───────────────────────────────────────────────────────────────
@app.post("/analyze")
def analyze():
    """
    Analyze adaptive quiz CSV and return JSON with student risk/confidence.
    Expected payload:
      {
        "schema": [...],
        "csv": "userid,username,quizname,difficultysum,...",
        "run_label": "manual_2025-10-24",
        "dryrun": true/false
      }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        csv_text = (data.get("csv") or "").strip()
        schema = data.get("schema") or ["userid","username","quizname","difficultysum","standarderror","measure","timetaken"]
        dryrun = bool(data.get("dryrun", False))
        run_label = data.get("run_label") or f"manual_{time.strftime('%Y-%m-%d')}"

        # --- Validation ---
        if not csv_text:
            return jsonify({"error": "Missing CSV data"}), 400

        # --- Dry-run mode (for testing without token) ---
        if dryrun or not OPENAI_API_KEY:
            lines = [ln for ln in csv_text.splitlines() if ln.strip()]
            hdr = lines[0].split(",") if lines else []
            items = []
            for row in lines[1:]:
                cols = row.split(",")
                rec = dict(zip(hdr, cols))
                uid = int(rec.get("userid", "0") or 0)
                items.append({
                    "userid": uid,
                    "risk_score": 50.0,
                    "confidence": 0.4,
                    "drivers": ["dry-run mode"],
                    "student_msg": "Dry-run preview.",
                    "teacher_msg": "Dry-run: Verify Moodle ↔ Analytics link.",
                    "features": rec
                })
            return jsonify({"run_label": run_label, "items": items})

        # --- Real LLM analysis ---
        schema_list = ", ".join(schema)
        prompt = f"""
You are a learning analytics model. Analyze this CSV and return JSON only.
Columns: {schema_list}.
Each record = 1 quiz attempt. Aggregate by userid.

Output exactly:
{{
  "run_label": "{run_label}",
  "items": [
    {{
      "userid": int,
      "risk_score": float,     # 0–100 (higher = higher risk)
      "confidence": float,     # 0–1 (model certainty)
      "drivers": [string],     # 1–4 short causes
      "student_msg": string,   # actionable note for student
      "teacher_msg": string    # actionable note for teacher
    }}
  ]
}}

If data insufficient, set confidence low (~0.3) and neutral risk (~50).
Return ONLY valid JSON — no markdown, no commentary.

CSV data:
{csv_text}
        """.strip()

        # --- Call OpenAI ---
        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": "You are a JSON-only learning analytics engine."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        # --- Parse response safely ---
        text = ""
        try:
            text = resp.output[0].content[0].text
        except Exception:
            text = getattr(resp, "output_text", "")

        try:
            parsed = json.loads(text)
        except Exception:
            # Try recovering from partial JSON
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(text[start:end+1])
            else:
                raise ValueError("Invalid JSON returned from model")

        parsed.setdefault("run_label", run_label)
        if not isinstance(parsed.get("items"), list):
            parsed["items"] = []

        return jsonify(parsed)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        payload = {"error": f"{type(e).__name__}: {e}"}
        if DEBUG_ERRORS:
            payload["trace"] = tb
        return jsonify(payload), 500

# ───────────────────────────────────────────────────────────────
# 🚀 Local runner (Railway will override PORT automatically)
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
