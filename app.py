# app.py — Analytics microservice for Moodle Adaptive Quiz
import os, json, time, traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import csv
from io import StringIO

# ───────────────────────────────────────────────────────────────
# ⚙️  Configuration
# ───────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "on").strip().lower() == "on"

app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=OPENAI_API_KEY)

def to_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

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
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        csv_text = (data.get("csv") or "").strip()
        schema = data.get("schema") or [
            "userid", "username", "quizname",
            "difficultysum", "standarderror", "measure", "timetaken"
        ]
        dryrun = bool(data.get("dryrun", False))
        run_label = data.get("run_label") or f"manual_{time.strftime('%Y-%m-%d')}"

        if not csv_text:
            return jsonify({"error": "Missing CSV data"}), 400

        rows = list(csv.DictReader(StringIO(csv_text)))
        by_user = {}

        for r in rows:
            uid = int(r.get("userid", 0) or 0)
            by_user.setdefault(uid, []).append(r)

        student_summaries = []
        for uid, attempts in by_user.items():
            n = len(attempts)
            measures = [to_float(a.get("measure")) for a in attempts]
            ses = [to_float(a.get("standarderror")) for a in attempts]
            times = [to_float(a.get("timetaken")) for a in attempts]
            diffs = [to_float(a.get("difficultysum")) for a in attempts]

            avg_measure = sum(measures) / max(n, 1)
            avg_se = sum(ses) / max(n, 1)
            avg_time = sum(times) / max(n, 1)
            avg_diff = sum(diffs) / max(n, 1)

            trend = 0.0
            if n >= 2:
                trend = measures[-1] - measures[0]

            student_summaries.append({
                "userid": uid,
                "attempt_count": n,
                "avg_measure": round(avg_measure, 3),
                "avg_standarderror": round(avg_se, 3),
                "avg_timetaken": round(avg_time, 1),
                "avg_difficultysum": round(avg_diff, 3),
                "measure_trend": round(trend, 3)
            })

        if dryrun or not OPENAI_API_KEY:
            items = []
            for s in student_summaries:
                items.append({
                    "userid": s["userid"],
                    "risk_score": 50.0,
                    "confidence": 0.4,
                    "drivers": ["dry-run mode"],
                    "student_msg": "Dry-run preview.",
                    "teacher_msg": "Dry-run: Verify Moodle ↔ Analytics link."
                })
            return jsonify({"run_label": run_label, "items": items})

        prompt = f"""
You are a learning analytics model. Analyze these per-student summaries and return JSON only.

Output exactly:
{{
  "run_label": "{run_label}",
  "items": [
    {{
      "userid": int,
      "risk_score": float,
      "confidence": float,
      "drivers": [string],
      "student_msg": string,
      "teacher_msg": string
    }}
  ]
}}

Use the summaries below to produce a differentiated risk score for each userid.
Compare students relative to one another.
Do not return the same risk score for all students unless their summaries are actually identical.
If confidence is low, still provide a best-effort score and at least one concrete driver.
Risk score must be between 0 and 100 inclusive. Never return a negative value. Use 85, not 0.85.
Return ONLY valid JSON.
Please avoid using any analytic terms for student and teacher message

Student summaries:
{json.dumps(student_summaries, ensure_ascii=False)}
        """.strip()

        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": "You are a JSON-only learning analytics engine."},
                {"role": "user", "content": prompt}
            ]
        )

        text = getattr(resp, "output_text", "")

        try:
            parsed = json.loads(text)
        except Exception:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(text[start:end+1])
            else:
                raise ValueError("Invalid JSON returned from model")
        parsed.setdefault("run_label", run_label)
        if not isinstance(parsed.get("items"), list):
            parsed["items"] = []

        for item in parsed["items"]:
            risk = item.get("risk_score")
        
            try:
                risk = float(risk)
            except (TypeError, ValueError):
                risk = 50.0
        
            # If model returns 0–1 scale, convert to 0–100
            if 0 <= risk <= 1:
                risk = risk * 100
        
            # Clamp to valid range
            risk = max(0.0, min(100.0, risk))
        
            item["risk_score"] = round(risk, 1) 
            
        return jsonify(parsed)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        payload = {"error": f"{type(e).__name__}: {e}"}
        if DEBUG_ERRORS:
            payload["trace"] = tb
        return jsonify(payload), 500



# ───────────────────────────────────────────────────────────────
# 🚀 Local runner (Railway overrides PORT automatically)
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
