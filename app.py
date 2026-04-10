# app.py — Analytics microservice for Moodle Adaptive Quiz
import os, json, time, traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import csv
from io import StringIO

# ───────────────────────────────────────────────────────────────
# ⚙️ Configuration
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

def normalize_risk_value(risk):
    try:
        risk = float(risk)
    except (TypeError, ValueError):
        risk = 50.0

    # If model returns 0–1 scale, convert to 0–100
    if 0 <= risk <= 1:
        risk = risk * 100

    return round(max(0.0, min(100.0, risk)), 1)

def risk_is_likely_inverted(items, summaries):
    by_user = {s["userid"]: s for s in summaries}
    pairs = [i for i in items if i.get("userid") in by_user]

    if len(pairs) < 2:
        return False

    def measure_of(item):
        return by_user[item["userid"]]["avg_measure"]

    strongest = max(pairs, key=measure_of)
    weakest = min(pairs, key=measure_of)

    strong_measure = by_user[strongest["userid"]]["avg_measure"]
    weak_measure = by_user[weakest["userid"]]["avg_measure"]

    strong_attempts = by_user[strongest["userid"]]["attempt_count"]
    weak_attempts = by_user[weakest["userid"]]["attempt_count"]

    strong_risk = normalize_risk_value(strongest.get("risk_score", 50))
    weak_risk = normalize_risk_value(weakest.get("risk_score", 50))

    return (
        strong_measure > weak_measure
        and strong_attempts >= weak_attempts
        and strong_risk > weak_risk + 15
    )

def parse_json_from_model_text(text, retry_label="model"):
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise ValueError(f"Invalid JSON returned from {retry_label}")

# ───────────────────────────────────────────────────────────────
# 🩺 Health endpoint
# ───────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return jsonify({"status": "ok", "model": MODEL})

# ───────────────────────────────────────────────────────────────
# 🔍 /analyze endpoint
# ───────────────────────────────────────────────────────────────
@app.post("/analyze")
def analyze():
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

        friendly_summaries = []
        for s in student_summaries:
            friendly_summaries.append({
                "userid": s["userid"],
                "quiz_attempt_count": s["attempt_count"],
                "average_ability_estimate": s["avg_measure"],
                "average_uncertainty": s["avg_standarderror"],
                "average_completion_time_seconds": s["avg_timetaken"],
                "average_question_difficulty": s["avg_difficultysum"],
                "ability_trend": s["measure_trend"],
                "interpretation_note": (
                    "Higher average_ability_estimate means stronger quiz performance and should reduce risk. "
                    "More quiz attempts usually reduce risk. "
                    "High uncertainty or harder questions alone should not make a strong student high-risk."
                )
            })

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

Risk score meaning:
- 0 to 20 = very low risk
- 21 to 40 = low risk
- 41 to 60 = moderate risk
- 61 to 80 = high risk
- 81 to 100 = very high risk

Important scoring rules:
1. Higher performance must LOWER risk.
2. Lower performance must RAISE risk.
3. More quiz attempts usually LOWER risk.
4. Fewer quiz attempts usually RAISE risk.
5. Improving performance over time should LOWER risk slightly.
6. Declining performance over time should RAISE risk slightly.
7. High uncertainty alone must NOT make a strong student high-risk.
8. Harder attempted questions alone must NOT make a strong student high-risk.
9. Use absolute educational meaning first. Use cohort comparison only as a secondary check.
10. If one student is clearly performing better than another student and has similar or better engagement, the stronger student should not receive a higher risk score.
11. Students with clearly strong performance and adequate attempts should usually score below 35.
12. Students with clearly weak performance and limited attempts should usually score above 65.

Important:
- Do not reverse the direction of performance.
- Do not treat strong students as high-risk.
- Do not give nearly identical scores to clearly different students.
- Risk score must be between 0 and 100 inclusive. Use 85, not 0.85.
- Confidence must be between 0 and 1.
- Return ONLY valid JSON.

Do not use technical field names anywhere in drivers, student_msg, or teacher_msg.

Student messages must be:
- written in plain English
- specific and actionable
- supportive but not generic
- 1 to 2 sentences only
- based on the student's likely learning behaviour

Teacher messages must be:
- practical and classroom-friendly
- focused on what the teacher can do next
- 1 to 2 sentences only
- written in plain English

Avoid vague phrases such as:
- "keep trying"
- "focus more on"
- "limited engagement"
- "inconsistent performance"

Instead, explain what seems to be happening and suggest one next step.

Student summaries:
{json.dumps(friendly_summaries, ensure_ascii=False)}
""".strip()

        resp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": "You are a JSON-only learning analytics engine."},
                {"role": "user", "content": prompt}
            ]
        )

        text = getattr(resp, "output_text", "")
        parsed = parse_json_from_model_text(text, "model")

        parsed.setdefault("run_label", run_label)
        if not isinstance(parsed.get("items"), list):
            parsed["items"] = []

        for item in parsed["items"]:
            item["risk_score"] = normalize_risk_value(item.get("risk_score"))

        if risk_is_likely_inverted(parsed["items"], student_summaries):
            repair_prompt = f"""
The previous output likely inverted the risk ordering.

Correct the scores using these rules:
- higher performance must lower risk
- lower performance must raise risk
- more attempts usually lower risk
- fewer attempts usually raise risk
- high uncertainty alone must not make a strong student high-risk

Return the same JSON format again, fully corrected.

Student summaries:
{json.dumps(friendly_summaries, ensure_ascii=False)}
""".strip()

            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": "You are a JSON-only learning analytics engine."},
                    {"role": "user", "content": repair_prompt}
                ]
            )

            text = getattr(resp, "output_text", "")
            parsed = parse_json_from_model_text(text, "model on retry")

            parsed.setdefault("run_label", run_label)
            if not isinstance(parsed.get("items"), list):
                parsed["items"] = []

            for item in parsed["items"]:
                item["risk_score"] = normalize_risk_value(item.get("risk_score"))

        term_map = {
            "avg_measure": "recent performance",
            "avg_standarderror": "uncertainty in performance",
            "avg_timetaken": "completion time",
            "avg_difficultysum": "difficulty of attempted questions",
            "measure_trend": "performance trend",
            "attempt_count": "number of attempts",
            "recent_performance_level": "recent performance",
            "performance_uncertainty": "uncertainty in performance",
            "average_completion_time": "completion time",
            "difficulty_of_attempted_questions": "difficulty of attempted questions",
            "performance_change_over_time": "performance trend",
            "number_of_attempts": "number of attempts",
            "quiz_attempt_count": "number of attempts",
            "average_ability_estimate": "recent performance",
            "average_uncertainty": "uncertainty in performance",
            "average_completion_time_seconds": "completion time",
            "average_question_difficulty": "difficulty of attempted questions",
            "ability_trend": "performance trend"
        }

        def sanitize_text(text):
            if not isinstance(text, str):
                return text
            for old, new in term_map.items():
                text = text.replace(old, new)
            return text

        for item in parsed["items"]:
            item["student_msg"] = sanitize_text(item.get("student_msg"))
            item["teacher_msg"] = sanitize_text(item.get("teacher_msg"))

            if isinstance(item.get("drivers"), list):
                cleaned = []
                for d in item["drivers"]:
                    if isinstance(d, str):
                        cleaned.append(sanitize_text(d))
                item["drivers"] = cleaned

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
