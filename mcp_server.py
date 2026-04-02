"""
Neulearn — MCP Server
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import AzureOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

mcp = FastMCP("neulearn-tools")

# Azure OpenAI client for LLM reasoning in tools
llm_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)


def get_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def _infer_semantic_fields_with_llm(profile: dict, episodes: list) -> dict:
    """
    Use LLM to infer learning_goal, preferred_explanation_style, and common_question_themes.
    Internal helper for compute_semantic_profile_update.
    """
    try:
        # Extract context for LLM
        mentor_exchanges = profile.get("_agent_context", {}).get("mentor_exchanges", [])
        declared_interests = profile.get("identity", {}).get("declared_interests", [])
        current_learning_goal = profile.get("identity", {}).get("learning_goal")
        current_explanation_style = profile.get("mentor_context", {}).get("preferred_explanation_style", "adaptive")
        current_question_themes = profile.get("mentor_context", {}).get("common_question_themes", [])
        
        # Build prompt for LLM
        prompt = f"""Analyze this learner's profile and infer missing semantic fields.

**Declared Interests:** {', '.join(declared_interests) if declared_interests else 'None'}
**Current Learning Goal:** {current_learning_goal or 'Not set'}
**Mentor Exchanges (recent):** {len(mentor_exchanges)}
{chr(10).join(f"- {ex}" for ex in mentor_exchanges[:5])}

**Episodes Summary:**
- Total episodes: {len(episodes)}
- Types: {', '.join(set(ep['type'] for ep in episodes))}

**Task:**
1. **learning_goal** (only if currently None): Infer specific technical goal from episodes and exchanges (e.g., "build production RAG systems"). Return null if insufficient data.

2. **preferred_explanation_style**: Analyze mentor exchanges. Return one of:
   - "analogy_based": Uses analogies, metaphors
   - "technical": Prefers precise specs, math, formulas
   - "visual": Requests diagrams, visualizations
   - "adaptive": Not enough data (need 5+ exchanges)

3. **common_question_themes**: Extract top 3 recurring themes from exchanges (max 3). Examples: "architecture comparisons", "debugging strategies", "when to use what".

**Output JSON only:**
```json
{{
  "learning_goal": "inferred goal or null",
  "preferred_explanation_style": "style",
  "common_question_themes": ["theme1", "theme2", "theme3"]
}}
```"""
        
        response = llm_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": "You are a learning analytics assistant. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        inferred = json.loads(result_text)
        
        # Apply inferences
        result = {
            "learning_goal": inferred.get("learning_goal") if not current_learning_goal else current_learning_goal,
            "preferred_explanation_style": inferred.get("preferred_explanation_style", current_explanation_style),
            "common_question_themes": inferred.get("common_question_themes", current_question_themes)
        }
        
        log.info(f"LLM inferred semantic fields: {result}")
        return result
        
    except Exception as e:
        log.error(f"LLM inference error: {e}")
        # Return safe defaults
        return {
            "learning_goal": current_learning_goal,
            "preferred_explanation_style": current_explanation_style,
            "common_question_themes": current_question_themes
        }



@mcp.tool()
def fetch_raw_self_assessment(user_id: str, session_id: str) -> str:
    log.info(f"Fetching raw self-assessment for user={user_id}, session={session_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, user_id, session_id,
                       declared_interests, self_assessment, created_at
                FROM   raw_self_assessment
                WHERE  user_id    = %s
                  AND  session_id = %s
                  AND  processed_at IS NULL
                LIMIT  1
            """, (user_id, session_id))
            row = cur.fetchone()
            if not row:
                log.debug(f"No unprocessed self-assessment found for user={user_id}, session={session_id}")
                return json.dumps({})
            log.debug(f"SQL returned: {dict(row)}")
            result = dict(row)
            result["created_at"] = result["created_at"].isoformat()
            log.info(f"Successfully fetched self-assessment for user={user_id}")
            return json.dumps(result)
    finally:
        conn.close()


@mcp.tool()
def fetch_raw_pre_assessment(user_id: str, session_id: str) -> str:
    log.info(f"Fetching raw pre-assessment for user={user_id}, session={session_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, user_id, session_id, domain, topic_id,
                       question_id, question_text,
                       selected_answer, correct_answer, is_correct
                FROM   raw_pre_assessment
                WHERE  user_id    = %s
                  AND  session_id = %s
                  AND  processed_at IS NULL
                ORDER  BY created_at
            """, (user_id, session_id))
            rows = cur.fetchall()
            if not rows:
                log.debug(f"No unprocessed pre-assessment found for user={user_id}, session={session_id}")
                return json.dumps({})
            log.debug(f"SQL returned {len(rows)} pre-assessment rows")
            result = {"user_id": user_id, "session_id": session_id,
                      "domains": {}, "row_ids": []}
            for r in rows:
                domain = r["domain"]
                if domain not in result["domains"]:
                    result["domains"][domain] = []
                result["domains"][domain].append({
                    "row_id":          r["id"],
                    "topic_id":        r["topic_id"],
                    "question_id":     r["question_id"],
                    "question_text":   r["question_text"],
                    "selected_answer": r["selected_answer"],
                    "correct_answer":  r["correct_answer"],
                    "is_correct":      r["is_correct"],
                })
                result["row_ids"].append(r["id"])
            return json.dumps(result)
    finally:
        conn.close()


def fetch_self_assessment_ratings(user_id: str, session_id: str) -> str:
    """Internal helper — not exposed as an MCP tool.
    Calibration is computed server-side inside compute_pre_assessment_scores."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT declared_interests, self_assessment
                FROM   raw_self_assessment
                WHERE  user_id    = %s
                  AND  session_id = %s
                LIMIT  1
            """, (user_id, session_id))
            row = cur.fetchone()
            return json.dumps(dict(row) if row else {})
    finally:
        conn.close()


@mcp.tool()
def compute_pre_assessment_scores(user_id: str, session_id: str) -> str:
    log.info(f"Computing pre-assessment scores for user={user_id}, session={session_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # Ordering guard
            cur.execute("""
                SELECT 1
                FROM   raw_self_assessment
                WHERE  user_id    = %s
                  AND  session_id = %s
                  AND  processed_at IS NOT NULL
                LIMIT  1
            """, (user_id, session_id))
            if not cur.fetchone():
                return json.dumps({
                    "status": "error",
                    "detail": (
                        "SELF_ASSESSMENT has not been processed for this "
                        "user+session. Run SELF_ASSESSMENT extraction first."
                    ),
                })

            cur.execute("""
                SELECT id, domain, topic_id, is_correct
                FROM   raw_pre_assessment
                WHERE  user_id    = %s
                  AND  session_id = %s
                  AND  processed_at IS NULL
            """, (user_id, session_id))
            rows = cur.fetchall()
            if not rows:
                return json.dumps({})

            cur.execute("""
                SELECT self_assessment
                FROM   raw_self_assessment
                WHERE  user_id    = %s
                  AND  session_id = %s
                LIMIT  1
            """, (user_id, session_id))
            sa_row = cur.fetchone()
            self_assessment = sa_row["self_assessment"] if sa_row else {}

        # Compute mcq_scores
        counts = {}
        row_ids = []
        for r in rows:
            d, t = r["domain"], r["topic_id"]
            counts.setdefault(d, {}).setdefault(t, {"correct": 0, "total": 0})
            counts[d][t]["total"] += 1
            if r["is_correct"]:
                counts[d][t]["correct"] += 1
            row_ids.append(r["id"])

        mcq_scores = {
            d: {
                t: {"score": v["correct"], "out_of": v["total"]}
                for t, v in topics.items()
            }
            for d, topics in counts.items()
        }

        # Compute overall_entry_level
        overall_entry_level = {}
        for d, topics in counts.items():
            total_correct = sum(v["correct"] for v in topics.values())
            total_q       = sum(v["total"]   for v in topics.values())
            pct = total_correct / total_q if total_q else 0
            overall_entry_level[d] = (
                "advanced"     if pct >= 0.75 else
                "intermediate" if pct >= 0.50 else
                "beginner"
            )

        # Compute calibration_delta
        calibration_delta = {}
        for d, topics in counts.items():
            calibration_delta[d] = {}
            sa_domain = self_assessment.get(d, {})
            for t, v in topics.items():
                pct         = v["correct"] / v["total"] if v["total"] else 0
                self_rating = sa_domain.get(t, "unknown")
                if self_rating in ("intermediate", "advanced") and pct < 0.5:
                    delta = "overconfident"
                elif self_rating in ("none", "beginner") and pct >= 0.75:
                    delta = "underconfident"
                elif self_rating == "unknown":
                    delta = "unknown"
                else:
                    delta = "accurate"
                calibration_delta[d][t] = delta

        return json.dumps({
            "mcq_scores":          mcq_scores,
            "overall_entry_level": overall_entry_level,
            "calibration_delta":   calibration_delta,
            "row_ids":             row_ids,
        })

    except Exception as e:
        log.error("compute_pre_assessment_scores error: %s", e)
        return json.dumps({"status": "error", "detail": str(e)})
    finally:
        conn.close()


@mcp.tool()
def compute_self_assessment_ratings(self_assessment_json: str) -> str:
    log.info("Computing self-assessment ratings")
    try:
        self_assessment = json.loads(self_assessment_json)
    except Exception as e:
        log.error(f"Invalid JSON in compute_self_assessment_ratings: {e}")
        return json.dumps({"status": "error", "detail": f"Invalid JSON: {e}"})

    if not isinstance(self_assessment, dict):
        return json.dumps({"status": "error",
                           "detail": "self_assessment_json must be a JSON object"})
    valid_levels = {"none", "beginner", "intermediate", "advanced"}
    for domain, topics in self_assessment.items():
        if not isinstance(topics, dict):
            return json.dumps({"status": "error",
                               "detail": f"domain '{domain}' must map to an object"})
        for topic_id, level in topics.items():
            if level not in valid_levels:
                log.warning("Unknown level '%s' for domain=%s topic=%s — treating as 'none'",
                            level, domain, topic_id)

    level_rank = {"none": 0, "beginner": 1, "intermediate": 2, "advanced": 3}
    ratings = {}

    for domain, topics in self_assessment.items():
        if not topics:
            ratings[domain] = "none"
            continue

        values = list(topics.values())
        total  = len(values)
        counts = {lvl: 0 for lvl in level_rank}
        for v in values:
            safe = v if v in counts else "none"
            counts[safe] += 1

        if counts["none"] == total:
            ratings[domain] = "none"
        elif counts["advanced"] > total / 2:
            ratings[domain] = "advanced"
        elif counts["intermediate"] > total / 2:
            ratings[domain] = "intermediate"
        elif counts["intermediate"] == 0 and counts["advanced"] == 0:
            ratings[domain] = "beginner"
        else:
            ratings[domain] = "intermediate"

    log.info(f"Successfully computed ratings for {len(ratings)} domains")
    return json.dumps(ratings)


@mcp.tool()
def write_episode(user_id: str, episode_type: str, schema_version: int, data_json: str) -> str:
    log.info(f"Writing episode type={episode_type} for user={user_id}, schema_version={schema_version}")
    conn = get_conn()
    try:
        valid_types = [
            "SELF_ASSESSMENT", "LEARNING_PATH_ASSIGNED", "COURSE_ACTIVITY",
            "MENTOR_CHAT", "COURSE_COMPLETED", "CAPSTONE_CODE_REVIEW",
            "CAPSTONE_TEST_RUN", "CAPSTONE_VIVA",
        ]
        if episode_type == "PRE_ASSESSMENT":
            return json.dumps({"status": "error",
                               "detail": "Use write_pre_assessment_episode for PRE_ASSESSMENT"})
        if episode_type not in valid_types:
            return json.dumps({"status": "error",
                               "detail": f"invalid type '{episode_type}'"})
        try:
            data = json.loads(data_json)
        except Exception as e:
            return json.dumps({"status": "error", "detail": f"data_json invalid JSON: {e}"})

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Idempotency guard - check session_id if present in data
            session_id_from_data = data.get("session_id")
            
            if session_id_from_data:
                # For session-based episodes: check user + type + session_id + time
                cur.execute("""
                    SELECT episode_id
                    FROM   episodic_episodes
                    WHERE  user_id   = %s
                      AND  type      = %s
                      AND  data->>'session_id' = %s
                      AND  timestamp > now() - interval '10 minutes'
                    LIMIT  1
                """, (user_id, episode_type, session_id_from_data))
            else:
                # For non-session episodes: check user + type + time only
                cur.execute("""
                    SELECT episode_id
                    FROM   episodic_episodes
                    WHERE  user_id   = %s
                      AND  type      = %s
                      AND  timestamp > now() - interval '10 minutes'
                    LIMIT  1
                """, (user_id, episode_type))
            
            existing = cur.fetchone()
            if existing:
                return json.dumps({
                    "episode_id": existing["episode_id"],
                    "status": "ok",
                    "note": "duplicate_blocked",
                })

            ep_id = str(uuid.uuid4())
            ts    = datetime.now(timezone.utc).isoformat()

            cur.execute("""
                INSERT INTO episodic_episodes
                    (episode_id, user_id, type, schema_version, timestamp, data)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (ep_id, user_id, episode_type, schema_version, ts, json.dumps(data)))
        conn.commit()
        log.info(f"Successfully wrote episode type={episode_type}, episode_id={ep_id} for user={user_id}")
        return json.dumps({"episode_id": ep_id, "status": "ok"})
    except Exception as e:
        conn.rollback()
        log.error(f"write_episode error for user={user_id}, type={episode_type}: {e}")
        return json.dumps({"status": "error", "detail": str(e)})
    finally:
        conn.close()


@mcp.tool()
def write_pre_assessment_episode(user_id: str, mcq_scores_json: str,
                                  overall_entry_level_json: str, calibration_delta_json: str) -> str:
    log.info(f"Writing pre-assessment episode for user={user_id}")
    conn = get_conn()
    try:
        try:
            data = {
                "mcq_scores":          json.loads(mcq_scores_json),
                "overall_entry_level": json.loads(overall_entry_level_json),
                "calibration_delta":   json.loads(calibration_delta_json),
            }
        except Exception as e:
            log.error(f"Invalid JSON in write_pre_assessment_episode for user={user_id}: {e}")
            return json.dumps({"status": "error", "detail": f"Invalid JSON arg: {e}"})

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT episode_id
                FROM   episodic_episodes
                WHERE  user_id   = %s
                  AND  type      = 'PRE_ASSESSMENT'
                  AND  timestamp > now() - interval '10 minutes'
                LIMIT  1
            """, (user_id,))
            existing = cur.fetchone()
            if existing:
                return json.dumps({
                    "episode_id": existing["episode_id"],
                    "status": "ok",
                    "note": "duplicate_blocked",
                })

            ep_id = str(uuid.uuid4())
            ts    = datetime.now(timezone.utc).isoformat()

            cur.execute("""
                INSERT INTO episodic_episodes
                    (episode_id, user_id, type, schema_version, timestamp, data)
                VALUES (%s, %s, 'PRE_ASSESSMENT', 1, %s, %s)
            """, (ep_id, user_id, ts, json.dumps(data)))
        conn.commit()
        log.info(f"Successfully wrote pre-assessment episode_id={ep_id} for user={user_id}")
        return json.dumps({"episode_id": ep_id, "status": "ok"})
    except Exception as e:
        conn.rollback()
        log.error(f"write_pre_assessment_episode error for user={user_id}: {e}")
        return json.dumps({"status": "error", "detail": str(e)})
    finally:
        conn.close()


@mcp.tool()
def mark_self_assessment_processed(user_id: str, session_id: str, episode_id: str) -> str:
    log.info(f"Marking self-assessment processed: user={user_id}, session={session_id}, episode={episode_id}")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE raw_self_assessment
                SET    processed_at = now(), episode_id = %s
                WHERE  user_id      = %s
                  AND  session_id   = %s
                  AND  processed_at IS NULL
            """, (episode_id, user_id, session_id))
        conn.commit()
        log.info(f"Successfully marked self-assessment processed for user={user_id}")
        return "ok"
    except Exception as e:
        conn.rollback()
        log.error(f"mark_self_assessment_processed error: {e}")
        return f"error: {e}"
    finally:
        conn.close()


@mcp.tool()
def mark_pre_assessment_processed(row_ids_json: str, episode_id: str) -> str:
    log.info(f"Marking pre-assessment processed: episode={episode_id}")
    conn = get_conn()
    try:
        row_ids = json.loads(row_ids_json)
        log.debug(f"Processing {len(row_ids)} pre-assessment rows")
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE raw_pre_assessment
                SET    processed_at = now(), episode_id = %s
                WHERE  id = ANY(%s) AND processed_at IS NULL
            """, (episode_id, row_ids))
        conn.commit()
        log.info(f"Successfully marked {len(row_ids)} pre-assessment rows processed")
        return "ok"
    except Exception as e:
        conn.rollback()
        log.error(f"mark_pre_assessment_processed error: {e}")
        return f"error: {e}"
    finally:
        conn.close()


# =============================================================================
# COURSE ACTIVITY — READ
# =============================================================================

@mcp.tool()
def fetch_raw_course_session(user_id: str, session_id: str) -> str:
    """
    Read all unprocessed raw_course_session rows for a user+session.
    Groups events into a structured payload ready for compute_course_activity_summary.

    Returns JSON:
    {
      "user_id": str,
      "session_id": str,
      "course_id": str,
      "row_ids": [str],
      "events": {
        "session_start_at": ISO str,
        "session_end_at":   ISO str | null,
        "exercise_events":  [ { row_id, exercise_id, attempt_number, event_type,
                                score, test_cases_passed, test_cases_total,
                                topic_id, completion_percent, created_at } ],
        "topic_events":     [ { topic_id, completion_percent, created_at } ]
      },
      "raw_chat_messages": [ { role, message, topic_id } ]
    }
    Returns {} if no unprocessed rows found.
    Returns { "status": "error", "detail": "..." } on failure.
    """
    log.info(f"Fetching raw course session for user={user_id}, session={session_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, course_id, event_type, exercise_id, attempt_number,
                       score, test_cases_passed, test_cases_total,
                       topic_id, completion_percent, metadata, created_at
                FROM   raw_course_session
                WHERE  user_id    = %s
                  AND  session_id = %s
                  AND  processed_at IS NULL
                ORDER  BY created_at
            """, (user_id, session_id))
            rows = cur.fetchall()

        if not rows:
            log.debug(f"No unprocessed course session found for user={user_id}, session={session_id}")
            return json.dumps({})

        log.debug(f"SQL returned {len(rows)} course session events")
        course_id   = rows[0]["course_id"]
        row_ids     = []
        session_start_at = None
        session_end_at   = None
        exercise_events  = []
        topic_events     = []
        raw_chat_messages = []

        for r in rows:
            row_ids.append(r["id"])
            etype    = r["event_type"]
            meta     = r["metadata"] or {}
            created  = r["created_at"].isoformat()

            if etype == "session_start" and session_start_at is None:
                session_start_at = created

            if etype == "session_end":
                session_end_at = created

            # Exercise events
            if etype in ("exercise_attempt", "exercise_pass", "exercise_fail"):
                exercise_events.append({
                    "row_id":            r["id"],
                    "exercise_id":       r["exercise_id"],
                    "attempt_number":    r["attempt_number"],
                    "event_type":        etype,
                    "score":             r["score"],
                    "test_cases_passed": r["test_cases_passed"],
                    "test_cases_total":  r["test_cases_total"],
                    "topic_id":          r["topic_id"],
                    "completion_percent": float(r["completion_percent"] or 0),
                    "created_at":        created,
                })

            # Topic coverage events (non-exercise, non-session rows)
            if etype not in ("exercise_attempt", "exercise_pass", "exercise_fail",
                             "session_start", "session_end"):
                topic_events.append({
                    "topic_id":          r["topic_id"],
                    "completion_percent": float(r["completion_percent"] or 0),
                    "created_at":        created,
                })

            # Extract chat messages from metadata
            if "chat" in meta:
                chat = meta["chat"]
                raw_chat_messages.append({
                    "role":     chat.get("role", "unknown"),
                    "message":  chat.get("message", ""),
                    "topic_id": r["topic_id"],
                })

        # Final completion_percent — from session_end row, else max seen
        final_completion = 0.0
        for r in rows:
            if r["event_type"] == "session_end":
                final_completion = float(r["completion_percent"] or 0)
                break
        if final_completion == 0.0 and rows:
            final_completion = max(
                float(r["completion_percent"] or 0) for r in rows
            )

        return json.dumps({
            "user_id":    user_id,
            "session_id": session_id,
            "course_id":  course_id,
            "row_ids":    row_ids,
            "events": {
                "session_start_at": session_start_at,
                "session_end_at":   session_end_at,
                "exercise_events":  exercise_events,
                "topic_events":     topic_events,
                "final_completion_percent": final_completion,
            },
            "raw_chat_messages": raw_chat_messages,
        })

    except Exception as e:
        log.error("fetch_raw_course_session error: %s", e)
        return json.dumps({"status": "error", "detail": str(e)})
    finally:
        conn.close()


# ==============================================================================
# COURSE ACTIVITY — COMPUTE
# Fully server-side. Derives all structured episode fields except
# chatbot_interactions_summary, which the agent produces from raw_chat_messages.
# =============================================================================

@mcp.tool()
def compute_course_activity_summary(session_json: str) -> str:
    """
    Compute all deterministic fields for a COURSE_ACTIVITY episode.
    Input: full JSON string returned by fetch_raw_course_session.

    Returns JSON:
    {
      "course_id":             str,
      "session_id":            str,
      "duration_minutes":      int,
      "topics_covered":        [str],
      "ide_exercises": [
        {
          "exercise_d":          str,
          "attempts":             int,
          "final_score":          int | null,
          "test_cases_passed":    int | null,
          "test_cases_total":     int | null,
          "time_to_solve_minutes": int | null
        }
      ],
      "completion_percentage": float
    }
    Note: chatbot_interactions_summary is NOT included — agent derives it
    from raw_chat_messages using its own LLM reasoning.

    Returns { "status": "error", "detail": "..." } on failure.
    """
    log.info("Computing course activity summary")
    try:
        session = json.loads(session_json)
    except Exception as e:
        log.error(f"Invalid JSON in compute_course_activity_summary: {e}")
        return json.dumps({"status": "error", "detail": f"Invalid JSON: {e}"})

    try:
        events   = session.get("events", {})
        course_id  = session["course_id"]
        session_id = session["session_id"]

        # ── Duration ─────────────────────────────────────────────────────
        start_str = events.get("session_start_at")
        end_str   = events.get("session_end_at")
        duration_minutes = 0
        if start_str and end_str:
            from datetime import datetime, timezone
            fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
            def parse_ts(s):
                # handle both with and without microseconds
                for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
                    try:
                        return datetime.strptime(s, fmt)
                    except ValueError:
                        continue
                return None
            t_start = parse_ts(start_str)
            t_end   = parse_ts(end_str)
            if t_start and t_end:
                duration_minutes = max(1, int((t_end - t_start).total_seconds() / 60))

        # ── Topics covered ────────────────────────────────────────────────
        seen_topics = []
        seen_set    = set()
        # from topic_events
        for ev in events.get("topic_events", []):
            t = ev.get("topic_id")
            if t and t not in seen_set:
                seen_set.add(t)
                seen_topics.append(t)
        # from exercise_events
        for ev in events.get("exercise_events", []):
            t = ev.get("topic_id")
            if t and t not in seen_set:
                seen_set.add(t)
                seen_topics.append(t)

        # ── IDE exercises ─────────────────────────────────────────────────
        ex_events = events.get("exercise_events", [])

        # Group by exercise_id
        ex_groups = {}
        for ev in ex_events:
            eid = ev["exercise_id"]
            if eid not in ex_groups:
                ex_groups[eid] = []
            ex_groups[eid].append(ev)

        ide_exercises = []
        for eid, evs in ex_groups.items():
            # Sort by attempt_number
            evs_sorted = sorted(evs, key=lambda x: x.get("attempt_number") or 0)

            attempt_count = sum(
                1 for e in evs_sorted if e["event_type"] == "exercise_attempt"
            )

            # Pass event gives final score and test cases
            pass_ev = next(
                (e for e in evs_sorted if e["event_type"] == "exercise_pass"), None
            )
            final_score       = pass_ev["score"]             if pass_ev else None
            test_cases_passed = pass_ev["test_cases_passed"] if pass_ev else None
            test_cases_total  = pass_ev["test_cases_total"]  if pass_ev else None

            # Time to solve: first attempt → pass event
            time_to_solve = None
            first_attempt = next(
                (e for e in evs_sorted if e["event_type"] == "exercise_attempt"), None
            )
            if first_attempt and pass_ev:
                def parse_ts(s):
                    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
                        try:
                            from datetime import datetime
                            return datetime.strptime(s, fmt)
                        except ValueError:
                            continue
                    return None
                t1 = parse_ts(first_attempt["created_at"])
                t2 = parse_ts(pass_ev["created_at"])
                if t1 and t2:
                    time_to_solve = max(1, int((t2 - t1).total_seconds() / 60))

            ide_exercises.append({
                "exercise_id":           eid,
                "attempts":              attempt_count,
                "final_score":           final_score,
                "test_cases_passed":     test_cases_passed,
                "test_cases_total":      test_cases_total,
                "time_to_solve_minutes": time_to_solve,
            })

        # ── Completion percentage ─────────────────────────────────────────
        completion_percentage = events.get("final_completion_percent", 0.0)

        log.info(f"Successfully computed course activity: course={course_id}, topics={len(seen_topics)}, exercises={len(ide_exercises)}")
        return json.dumps({
            "course_id":             course_id,
            "session_id":            session_id,
            "duration_minutes":      duration_minutes,
            "topics_covered":        seen_topics,
            "ide_exercises":         ide_exercises,
            "completion_percentage": completion_percentage,
        })

    except Exception as e:
        log.error("compute_course_activity_summary error: %s", e)
        return json.dumps({"status": "error", "detail": str(e)})


# =============================================================================
# COURSE ACTIVITY — MARK PROCESSED
# =============================================================================

@mcp.tool()
def mark_course_session_processed(user_id: str, session_id: str, episode_id: str) -> str:
    """
    Mark all raw_course_session rows for a user+session as processed.
    Call only after write_episode returns status ok.
    Returns 'ok' or error string.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE raw_course_session
                SET    processed_at = now(), episode_id = %s
                WHERE  user_id      = %s
                  AND  session_id   = %s
                  AND  processed_at IS NULL
            """, (episode_id, user_id, session_id))
        conn.commit()
        log.info(
            "raw_course_session marked processed: user=%s session=%s episode=%s",
            user_id, session_id, episode_id,
        )
        return "ok"
    except Exception as e:
        conn.rollback()
        log.error("mark_course_session_processed error: %s", e)
        return f"error: {e}"
    finally:
        conn.close()


# =============================================================================
# MENTOR CHAT — FETCH RAW
# =============================================================================

@mcp.tool()
def fetch_raw_mentor_chat(user_id: str, session_id: str) -> str:
    """
    Read all unprocessed raw_mentor_chat_turns for a user+session.
    Groups turns into a structured payload ready for compute_mentor_chat_summary.

    Returns JSON:
    {
      "user_id": str,
      "session_id": str,
      "course_id": str | null,
      "pathway_id": str | null,
      "row_ids": [str],
      "turns": [
        {
          "turn_number": int,
          "role": "user" | "assistant",
          "message": str,
          "topic_id": str | null,
          "created_at": ISO str
        }
      ]
    }
    Returns {} if no unprocessed rows found.
    Returns { "status": "error", "detail": "..." } on failure.
    """
    log.info(f"Fetching raw mentor chat for user={user_id}, session={session_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, course_id, pathway_id, turn_number,
                       role, message, topic_id, created_at
                FROM   raw_mentor_chat_turns
                WHERE  user_id    = %s
                  AND  session_id = %s
                  AND  processed_at IS NULL
                ORDER  BY turn_number
            """, (user_id, session_id))
            rows = cur.fetchall()

        if not rows:
            log.debug(f"No unprocessed mentor chat found for user={user_id}, session={session_id}")
            return json.dumps({})

        log.debug(f"SQL returned {len(rows)} mentor chat turns")
        course_id   = rows[0]["course_id"]
        pathway_id  = rows[0]["pathway_id"]
        row_ids     = []
        turns       = []

        for r in rows:
            row_ids.append(r["id"])
            turns.append({
                "turn_number": r["turn_number"],
                "role":        r["role"],
                "message":     r["message"],
                "topic_id":    r["topic_id"],
                "created_at":  r["created_at"].isoformat(),
            })

        return json.dumps({
            "user_id":    user_id,
            "session_id": session_id,
            "course_id":  course_id,
            "pathway_id": pathway_id,
            "row_ids":    row_ids,
            "turns":      turns,
        })

    except Exception as e:
        log.error("fetch_raw_mentor_chat error: %s", e)
        return json.dumps({"status": "error", "detail": str(e)})
    finally:
        conn.close()


# =============================================================================
# MENTOR CHAT — COMPUTE SUMMARY
# =============================================================================

@mcp.tool()
def compute_mentor_chat_summary(session_json: str) -> str:
    """
    Compute all deterministic fields for a MENTOR_CHAT episode.
    Input: full JSON string returned by fetch_raw_mentor_chat.

    Returns JSON:
    {
      "session_id":        str,
      "course_id":         str | null,
      "pathway_id":        str | null,
      "duration_minutes":  int,
      "turn_count":        int,
      "topics_discussed":  [str]
    }
    Note: exchanges_summary, unresolved_questions, understanding_signals,
    and question_types are NOT included — agent derives them from turns
    using its own LLM reasoning.

    Returns { "status": "error", "detail": "..." } on failure.
    """
    log.info("Computing mentor chat summary")
    try:
        session = json.loads(session_json)
    except Exception as e:
        log.error(f"Invalid JSON in compute_mentor_chat_summary: {e}")
        return json.dumps({"status": "error", "detail": f"Invalid JSON: {e}"})

    try:
        turns       = session.get("turns", [])
        session_id  = session["session_id"]
        course_id   = session.get("course_id")
        pathway_id  = session.get("pathway_id")

        # ── Duration ─────────────────────────────────────────────────────
        duration_minutes = 0
        if len(turns) >= 2:
            first_ts = turns[0]["created_at"]
            last_ts  = turns[-1]["created_at"]
            
            from datetime import datetime
            def parse_ts(s):
                for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                           "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        return datetime.fromisoformat(s.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                return None
            
            t_first = parse_ts(first_ts)
            t_last  = parse_ts(last_ts)
            if t_first and t_last:
                duration_minutes = max(1, int((t_last - t_first).total_seconds() / 60))

        # ── Turn count ───────────────────────────────────────────────────
        turn_count = len(turns)

        # ── Topics discussed ──────────────────────────────────────────────
        seen_topics = []
        seen_set    = set()
        for turn in turns:
            t = turn.get("topic_id")
            if t and t not in seen_set:
                seen_set.add(t)
                seen_topics.append(t)

        log.info(f"Successfully computed mentor chat summary: turns={turn_count}, topics={len(seen_topics)}, duration={duration_minutes}min")
        return json.dumps({
            "session_id":        session_id,
            "course_id":         course_id,
            "pathway_id":        pathway_id,
            "duration_minutes":  duration_minutes,
            "turn_count":        turn_count,
            "topics_discussed":  seen_topics,
        })

    except Exception as e:
        log.error("compute_mentor_chat_summary error: %s", e)
        return json.dumps({"status": "error", "detail": str(e)})


# =============================================================================
# MENTOR CHAT — MARK PROCESSED
# =============================================================================

@mcp.tool()
def mark_mentor_chat_processed(user_id: str, session_id: str, episode_id: str) -> str:
    """
    Mark all raw_mentor_chat_turns rows for a user+session as processed.
    Call only after write_episode returns status ok.
    Returns 'ok' or error string.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE raw_mentor_chat_turns
                SET    processed_at = now(), episode_id = %s
                WHERE  user_id      = %s
                  AND  session_id   = %s
                  AND  processed_at IS NULL
            """, (episode_id, user_id, session_id))
        conn.commit()
        log.info(
            "raw_mentor_chat_turns marked processed: user=%s session=%s episode=%s",
            user_id, session_id, episode_id,
        )
        return "ok"
    except Exception as e:
        conn.rollback()
        log.error("mark_mentor_chat_processed error: %s", e)
        return f"error: {e}"
    finally:
        conn.close()


# =============================================================================
# COURSE_COMPLETED — FETCH, COMPUTE, MARK
# =============================================================================

@mcp.tool()
def fetch_raw_course_completion(user_id: str, course_id: str) -> str:
    """
    Fetch unprocessed course completion record and aggregate session data.
    Returns completion record + all session history for areas_of_struggle/strength computation.
    Returns {} if no unprocessed completion found.
    """
    log.info(f"Fetching raw course completion for user={user_id}, course={course_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get completion record
            cur.execute("""
                SELECT id, user_id, course_id, pathway_id, final_score, completed_at
                FROM   raw_course_completion
                WHERE  user_id = %s AND course_id = %s AND processed_at IS NULL
                LIMIT  1
            """, (user_id, course_id))
            completion = cur.fetchone()
            if not completion:
                log.debug(f"No unprocessed course completion found for user={user_id}, course={course_id}")
                return json.dumps({})
            
            log.debug(f"SQL returned completion: {dict(completion)}")
            # Get all session data for this course
            cur.execute("""
                SELECT session_id, course_id, event_type, exercise_id,
                       attempt_number, score, test_cases_passed, test_cases_total,
                       topic_id, metadata, created_at
                FROM   raw_course_session
                WHERE  user_id = %s AND course_id = %s
                ORDER  BY created_at
            """, (user_id, course_id))
            sessions = cur.fetchall()
            
            log.debug(f"SQL returned {len(sessions)} session events for course completion")
            result = {
                "completion_id": completion["id"],
                "user_id": completion["user_id"],
                "course_id": completion["course_id"],
                "pathway_id": completion["pathway_id"],
                "final_score": completion["final_score"],
                "completed_at": completion["completed_at"].isoformat(),
                "session_events": []
            }
            
            for s in sessions:
                result["session_events"].append({
                    "session_id": s["session_id"],
                    "event_type": s["event_type"],
                    "exercise_id": s["exercise_id"],
                    "attempt_number": s["attempt_number"],
                    "score": s["score"],
                    "test_cases_passed": s["test_cases_passed"],
                    "test_cases_total": s["test_cases_total"],
                    "topic_id": s["topic_id"],
                    "metadata": s["metadata"],
                    "created_at": s["created_at"].isoformat()
                })
            
            log.info(f"Successfully fetched course completion: user={user_id}, course={course_id}, sessions={len(sessions)}")
            return json.dumps(result)
    finally:
        conn.close()


@mcp.tool()
def compute_course_completed_summary(course_data_json: str) -> str:
    """
    Compute aggregated stats for COURSE_COMPLETED episode.
    Returns: total_sessions, avg_exercise_score, total_duration_hours,
             areas_of_struggle (topic_ids where score < 70 or attempts > 3),
             areas_of_strength (topic_ids where score > 85 on first attempt).
    """
    log.info("Computing course completed summary")
    try:
        data = json.loads(course_data_json)
        sessions = data.get("session_events", [])
        
        # Count unique sessions
        session_ids = set()
        total_duration_minutes = 0
        
        # Track exercises by topic
        topic_exercises = {}  # topic_id -> [{"exercise_id", "score", "attempts"}]
        
        for event in sessions:
            event_type = event.get("event_type")
            session_id = event.get("session_id")
            
            if session_id:
                session_ids.add(session_id)
            
            # Extract duration from session_end events
            if event_type == "session_end":
                metadata = event.get("metadata", {})
                duration_min = metadata.get("duration_min", 0)
                total_duration_minutes += duration_min
            
            # Track exercise passes for struggle/strength analysis
            if event_type == "exercise_pass":
                topic_id = event.get("topic_id")
                exercise_id = event.get("exercise_id")
                score = event.get("score", 0)
                attempt_number = event.get("attempt_number", 1)
                
                if topic_id:
                    if topic_id not in topic_exercises:
                        topic_exercises[topic_id] = []
                    topic_exercises[topic_id].append({
                        "exercise_id": exercise_id,
                        "score": score,
                        "attempts": attempt_number
                    })
        
        # Compute avg_exercise_score
        all_scores = []
        for exercises in topic_exercises.values():
            for ex in exercises:
                if ex["score"] is not None:
                    all_scores.append(ex["score"])
        avg_exercise_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0
        
        # Identify areas_of_struggle (score < 70 OR attempts > 3)
        areas_of_struggle = []
        for topic_id, exercises in topic_exercises.items():
            has_struggle = False
            for ex in exercises:
                if (ex["score"] and ex["score"] < 70) or (ex["attempts"] and ex["attempts"] > 3):
                    has_struggle = True
                    break
            if has_struggle:
                areas_of_struggle.append(topic_id)
        
        # Identify areas_of_strength (score > 85 AND first attempt)
        areas_of_strength = []
        for topic_id, exercises in topic_exercises.items():
            has_strength = False
            for ex in exercises:
                if ex["score"] and ex["score"] > 85 and ex["attempts"] == 1:
                    has_strength = True
                    break
            if has_strength and topic_id not in areas_of_struggle:
                areas_of_strength.append(topic_id)
        
        return json.dumps({
            "course_id": data.get("course_id"),
            "total_sessions": len(session_ids),
            "avg_exercise_score": avg_exercise_score,
            "total_duration_hours": round(total_duration_minutes / 60, 1),
            "final_assessment_score": data.get("final_score", 0),
            "areas_of_struggle": areas_of_struggle,
            "areas_of_strength": areas_of_strength
        })
        log.info(f"Successfully computed course completed summary: sessions={len(session_ids)}, avg_score={avg_exercise_score}")
        return result
    
    except Exception as e:
        log.error("compute_course_completed_summary error: %s", e)
        return json.dumps({"status": "error", "detail": str(e)})


@mcp.tool()
def mark_course_completion_processed(user_id: str, course_id: str, episode_id: str) -> str:
    """
    Mark raw_course_completion row as processed.
    Call only after write_episode succeeds.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE raw_course_completion
                SET    processed_at = now(), episode_id = %s
                WHERE  user_id = %s AND course_id = %s AND processed_at IS NULL
            """, (episode_id, user_id, course_id))
        conn.commit()
        log.info("raw_course_completion marked processed: user=%s course=%s episode=%s",
                 user_id, course_id, episode_id)
        return "ok"
    except Exception as e:
        conn.rollback()
        log.error("mark_course_completion_processed error: %s", e)
        return f"error: {e}"
    finally:
        conn.close()


# =============================================================================
# CAPSTONE_CODE_REVIEW — FETCH, COMPUTE, MARK
# =============================================================================

@mcp.tool()
def fetch_raw_code_review(user_id: str, capstone_id: str, attempt_id: str) -> str:
    """
    Fetch unprocessed code review output from GitLab agent.
    Returns review_output JSON with issues, scores, tech evaluations.
    Returns {} if no unprocessed review found.
    """
    log.info(f"Fetching raw code review for user={user_id}, capstone={capstone_id}, attempt={attempt_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, user_id, capstone_id, pathway_id, attempt_id,
                       repo_url, commit_sha, review_output, created_at
                FROM   raw_code_review
                WHERE  user_id = %s AND capstone_id = %s AND attempt_id = %s
                  AND  processed_at IS NULL
                LIMIT  1
            """, (user_id, capstone_id, attempt_id))
            row = cur.fetchone()
            if not row:
                log.debug(f"No unprocessed code review found for user={user_id}, capstone={capstone_id}")
                return json.dumps({})
            
            log.debug(f"SQL returned code review: {dict(row)}")
            result = dict(row)
            result["created_at"] = result["created_at"].isoformat()
            log.info(f"Successfully fetched code review for user={user_id}, capstone={capstone_id}")
            return json.dumps(result)
    finally:
        conn.close()


@mcp.tool()
def compute_code_review_summary(review_json: str) -> str:
    """
    Use LLM to extract structured tech evaluation data from code review output for CAPSTONE_CODE_REVIEW episode.
    Returns tech evaluations with DESCRIPTIVE understanding, design, code_quality, test_coverage, verdict per technology.
    """
    log.info("Computing code review summary with LLM")
    try:
        data = json.loads(review_json)
        review_output = data.get("review_output", {})
        
        # Extract key data for LLM analysis
        issues = review_output.get("issues", [])
        technology_breakdown = review_output.get("technology_breakdown", {})
        
        # Build prompt for LLM
        prompt = f"""Analyze this capstone code review and write DESCRIPTIVE evaluations for each technology.

**Overall Metrics:**
- Files reviewed: {review_output.get('files_reviewed', 0)}
- Score: {review_output.get('score', 0)}/100
- Critical issues: {sum(1 for i in issues if i.get('severity') in ['critical', 'high'])}
- Minor issues: {sum(1 for i in issues if i.get('severity') in ['medium', 'low'])}

**Issues by Technology:**
{json.dumps([{"technology": i.get("technology"), "severity": i.get("severity"), "category": i.get("category"), "comment": i.get("comment")[:150]} for i in issues[:15]], indent=2)}

**Technology Breakdown (if available):**
{json.dumps(technology_breakdown, indent=2) if technology_breakdown else "Not provided"}

**CRITICAL: Write 2-3 sentence DESCRIPTIVE text for each field. DO NOT use single words or short labels.**

For each technology mentioned in the issues, write:

1. **understanding**: Describe what they understood well and what they missed. Mention specific concepts, patterns, or architectures.
   Example: "Shows strong grasp of Django's MTV pattern with proper model-view separation and use of class-based views. However, demonstrates confusion about QuerySet lazy evaluation and N+1 query problems when fetching related objects."

2. **design**: Describe architectural strengths and weaknesses. Be specific about design choices.
   Example: "API follows RESTful conventions with proper resource naming and HTTP verb usage. Authentication flow is well-structured using JWT tokens. However, missing rate limiting middleware and no strategy for handling token refresh or revocation."

3. **code_quality**: Describe implementation quality, patterns used, maintainability.
   Example: "Code follows PEP-8 style guidelines and uses descriptive variable names. Good use of type hints for function signatures. However, many functions exceed 50 lines with mixed responsibilities, and error handling relies on bare except clauses that swallow specific exceptions."

4. **test_coverage**: Describe what's tested and what's missing. Be specific about gaps.
   Example: "Comprehensive unit tests for business logic with good use of fixtures and parametrized tests. However, integration tests only cover happy paths. Missing tests for authentication failures, database constraint violations, and edge cases like empty request bodies."

5. **verdict**: pass OR needs_revision (just the label)

Dont invent Any answers on your own. Evaluate only on the actual code review. 
If data is not available you dont need to create your own data.

**Output JSON only:**
```json
{{
  "tech_evaluations": [
    {{
      "technology": "django",
      "understanding": "Shows strong grasp of Django MTV pattern but confused about QuerySet lazy evaluation",
      "design": "API follows REST conventions well but missing rate limiting strategy",
      "code_quality": "Good PEP-8 compliance and type hints but functions too long with mixed responsibilities",
      "test_coverage": "Comprehensive unit tests but integration tests only cover happy paths",
      "verdict": "needs_revision"
    }}
  ]
}}
```"""
        
        response = llm_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": "You are a senior code reviewer. Write detailed, descriptive evaluations (2-3 sentences per field). Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        llm_result = json.loads(result_text)
        
        # Count issues by severity
        critical_count = sum(1 for i in issues if i.get("severity") in ["critical", "high"])
        minor_count = sum(1 for i in issues if i.get("severity") in ["medium", "low"])
        
        # Determine overall verdict
        passed = review_output.get("passed", False)
        if critical_count > 0:
            overall_verdict = "needs_revision"
        elif passed:
            overall_verdict = "pass"
        else:
            overall_verdict = "fail"
        
        result = {
            "capstone_id": data.get("capstone_id"),
            "submission_id": data.get("commit_sha"),
            "timeline_adherence": "on_time",
            "overall_issues_critical": critical_count,
            "overall_issues_minor": minor_count,
            "overall_verdict": overall_verdict,
            "tech_evaluations": llm_result.get("tech_evaluations", [])
        }
        
        log.info(f"LLM computed code review summary: verdict={overall_verdict}, tech_evals={len(result['tech_evaluations'])}")
        return json.dumps(result)
    
    except Exception as e:
        log.error("compute_code_review_summary error: %s", e, exc_info=True)
        return json.dumps({"status": "error", "detail": str(e)})


@mcp.tool()
def mark_code_review_processed(user_id: str, capstone_id: str, attempt_id: str, episode_id: str) -> str:
    """
    Mark raw_code_review row as processed.
    Call only after write_episode succeeds.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE raw_code_review
                SET    processed_at = now(), episode_id = %s
                WHERE  user_id = %s AND capstone_id = %s AND attempt_id = %s
                  AND  processed_at IS NULL
            """, (episode_id, user_id, capstone_id, attempt_id))
        conn.commit()
        log.info("raw_code_review marked processed: user=%s capstone=%s episode=%s",
                 user_id, capstone_id, episode_id)
        return "ok"
    except Exception as e:
        conn.rollback()
        log.error("mark_code_review_processed error: %s", e)
        return f"error: {e}"
    finally:
        conn.close()


# =============================================================================
# CAPSTONE_TEST_RUN — FETCH, COMPUTE, MARK
# =============================================================================

@mcp.tool()
def fetch_raw_test_review(user_id: str, capstone_id: str, attempt_id: str) -> str:
    """
    Fetch unprocessed test review output from testing agent.
    Returns test_output JSON with test_results array, coverage.
    Returns {} if no unprocessed review found.
    """
    log.info(f"Fetching raw test review for user={user_id}, capstone={capstone_id}, attempt={attempt_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, user_id, capstone_id, pathway_id, attempt_id,
                       test_output, created_at
                FROM   raw_test_review
                WHERE  user_id = %s AND capstone_id = %s AND attempt_id = %s
                  AND  processed_at IS NULL
                LIMIT  1
            """, (user_id, capstone_id, attempt_id))
            row = cur.fetchone()
            if not row:
                log.debug(f"No unprocessed test review found for user={user_id}, capstone={capstone_id}")
                return json.dumps({})
            
            log.debug(f"SQL returned test review: {dict(row)}")
            result = dict(row)
            result["created_at"] = result["created_at"].isoformat()
            log.info(f"Successfully fetched test review for user={user_id}, capstone={capstone_id}")
            return json.dumps(result)
    finally:
        conn.close()


@mcp.tool()
def compute_test_run_summary(test_json: str) -> str:
    """
    Use LLM to extract RICH structured data from test output for CAPSTONE_TEST_RUN episode.
    Returns detailed analysis including failure patterns, coverage gaps, and verdict.
    """
    log.info("Computing test run summary with LLM")
    try:
        data = json.loads(test_json)
        test_output = data.get("test_output", {})
        
        test_results = test_output.get("test_results", [])
        tests_passed = test_output.get("passed", 0)
        tests_total = test_output.get("total_tests", 0)
        coverage_percent = test_output.get("coverage_percent", 0)
        
        # Extract failed tests with details
        failed_tests = [t for t in test_results if not t.get("passed", False)]
        
        # Build prompt for LLM
        prompt = f"""Analyze this capstone test run and extract failure patterns and coverage insights.

**Overall Metrics:**
- Total tests: {tests_total}
- Passed: {tests_passed}
- Failed: {len(failed_tests)}
- Coverage: {coverage_percent}%

**Failed Tests:**
{json.dumps([{"test_name": t.get("test_name"), "error": t.get("error_message")[:150]} for t in failed_tests[:10]], indent=2)}

**Test Categories (if available):**
{json.dumps(test_output.get("test_categories", {}), indent=2)}

**Coverage by Component (if available):**
{json.dumps(test_output.get("coverage_by_component", {}), indent=2)}

**Task:** Analyze the test failures and extract:
1. **failed_tests_detailed**: For each failed test, extract descriptive info about what failed and why
2. **failure_analysis**: Group failures by common root causes or patterns
3. **coverage_gaps**: Identify what's not being tested adequately

Dont invent Any answers on your own. Evaluate only on the actual test Review.
If data is not available you dont need to create your own data.

**Output JSON only:**
```json
{{
  "failed_tests_detailed": [
    {{
      "test_name": "test_user_authentication_expired_token",
      "category": "integration",
      "technology": "authentication",
      "root_cause": "Token expiration check not implemented in middleware",
      "impact": "Expired tokens accepted as valid, security vulnerability"
    }}
  ],
  "failure_analysis": {{
    "by_severity": {{
      "critical": ["test_sql_injection_user_input"],
      "high": ["test_password_reset_race_condition"]
    }},
    "by_technology": {{
      "authentication": ["test_user_authentication_expired_token", "test_password_reset_race_condition"],
      "database": ["test_sql_injection_user_input"]
    }},
    "common_root_causes": ["Input validation gaps", "Race condition in async handlers"]
  }},
  "coverage_gaps": [
    "No tests for session timeout behavior",
    "No tests for concurrent login attempts",
    "No load tests for API rate limiting"
  ]
}}
```"""
        
        response = llm_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": "You are a test analysis expert. Output valid JSON only with failure patterns and coverage gaps."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        llm_result = json.loads(result_text)
        
        # Determine verdict
        overall_passed = test_output.get("overall_passed", False)
        if overall_passed:
            agent_verdict = "pass"
        elif tests_passed / tests_total >= 0.7 if tests_total > 0 else False:
            agent_verdict = "partial_pass"
        else:
            agent_verdict = "fail"
        
        # Extract simple failed test IDs
        failed_test_ids = [t.get("test_name") for t in failed_tests]
        
        result = {
            "capstone_id": data.get("capstone_id"),
            "submission_id": data.get("attempt_id"),
            "timeline_adherence": "on_time",
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "coverage_percent": coverage_percent,
            "coverage_by_component": test_output.get("coverage_by_component", {}),
            "test_categories": test_output.get("test_categories", {}),
            "failed_test_ids": failed_test_ids,
            "failed_tests_detailed": llm_result.get("failed_tests_detailed", []),
            "failure_analysis": llm_result.get("failure_analysis", {}),
            "coverage_gaps": llm_result.get("coverage_gaps", []),
            "agent_verdict": agent_verdict
        }
        
        log.info(f"LLM computed test run summary: passed={tests_passed}/{tests_total}, verdict={agent_verdict}, failed_count={len(failed_test_ids)}")
        return json.dumps(result)
    
    except Exception as e:
        log.error("compute_test_run_summary error: %s", e, exc_info=True)
        return json.dumps({"status": "error", "detail": str(e)})


@mcp.tool()
def mark_test_review_processed(user_id: str, capstone_id: str, attempt_id: str, episode_id: str) -> str:
    """
    Mark raw_test_review row as processed.
    Call only after write_episode succeeds.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE raw_test_review
                SET    processed_at = now(), episode_id = %s
                WHERE  user_id = %s AND capstone_id = %s AND attempt_id = %s
                  AND  processed_at IS NULL
            """, (episode_id, user_id, capstone_id, attempt_id))
        conn.commit()
        log.info("raw_test_review marked processed: user=%s capstone=%s episode=%s",
                 user_id, capstone_id, episode_id)
        return "ok"
    except Exception as e:
        conn.rollback()
        log.error("mark_test_review_processed error: %s", e)
        return f"error: {e}"
    finally:
        conn.close()


# =============================================================================
# CAPSTONE_VIVA — FETCH, COMPUTE, MARK
# =============================================================================

@mcp.tool()
def fetch_raw_viva(user_id: str, capstone_id: str, attempt_id: str) -> str:
    """
    Fetch unprocessed viva turns for a capstone.
    Returns array of turns with role, message, turn_number.
    Returns {} if no unprocessed turns found.
    """
    log.info(f"Fetching raw viva turns for user={user_id}, capstone={capstone_id}, attempt={attempt_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Check if any turns exist and are unprocessed
            cur.execute("""
                SELECT COUNT(*) as cnt
                FROM   raw_viva_turns
                WHERE  user_id = %s AND capstone_id = %s AND attempt_id = %s
                  AND  processed_at IS NULL
            """, (user_id, capstone_id, attempt_id))
            count_row = cur.fetchone()
            if not count_row or count_row["cnt"] == 0:
                log.debug(f"No unprocessed viva turns found for user={user_id}, capstone={capstone_id}")
                return json.dumps({})
            
            log.debug(f"SQL found {count_row['cnt']} unprocessed viva turns")
            # Fetch all turns
            cur.execute("""
                SELECT id, user_id, session_id, capstone_id, pathway_id, attempt_id,
                       turn_number, role, message, created_at
                FROM   raw_viva_turns
                WHERE  user_id = %s AND capstone_id = %s AND attempt_id = %s
                  AND  processed_at IS NULL
                ORDER  BY turn_number
            """, (user_id, capstone_id, attempt_id))
            turns = cur.fetchall()
            
            log.debug(f"SQL returned {len(turns)} viva turns")
            result = {
                "user_id": user_id,
                "session_id": turns[0]["session_id"] if turns else None,
                "capstone_id": capstone_id,
                "pathway_id": turns[0]["pathway_id"] if turns else None,
                "attempt_id": attempt_id,
                "row_ids": [],
                "turns": []
            }
            
            for turn in turns:
                result["row_ids"].append(turn["id"])
                result["turns"].append({
                    "turn_number": turn["turn_number"],
                    "role": turn["role"],
                    "message": turn["message"],
                    "created_at": turn["created_at"].isoformat()
                })
            
            log.info(f"Successfully fetched viva turns: user={user_id}, capstone={capstone_id}, turns={len(turns)}")
            return json.dumps(result)
    finally:
        conn.close()


@mcp.tool()
def compute_viva_summary(viva_json: str) -> str:
    """
    Use LLM to compute FULL viva analysis for CAPSTONE_VIVA episode.
    Returns: capstone_id, viva_session_id, timeline_adherence, questions_asked, duration_minutes,
             answers_satisfactory, weak_areas_identified, strong_areas_identified, viva_score, agent_verdict.
    """
    log.info("Computing viva summary with LLM")
    try:
        data = json.loads(viva_json)
        turns = data.get("turns", [])
        
        # Count questions (viva_agent turns)
        questions_asked = sum(1 for t in turns if t.get("role") == "viva_agent")
        
        # Compute duration from first to last turn timestamp
        duration_minutes = 0
        if len(turns) >= 2:
            from datetime import datetime as dt
            first_ts = dt.fromisoformat(turns[0]["created_at"].replace('Z', '+00:00'))
            last_ts = dt.fromisoformat(turns[-1]["created_at"].replace('Z', '+00:00'))
            duration_minutes = round((last_ts - first_ts).total_seconds() / 60, 1)
        
        # Format Q&A pairs for LLM
        qa_pairs = []
        for i in range(len(turns) - 1):
            if turns[i].get("role") == "viva_agent" and turns[i+1].get("role") == "user":
                evaluator_feedback = ""
                if i + 2 < len(turns) and turns[i+2].get("role") == "viva_agent":
                    evaluator_feedback = turns[i+2].get("message", "")[:200]
                
                qa_pairs.append({
                    "question": turns[i].get("message", ""),
                    "answer": turns[i+1].get("message", ""),
                    "evaluator_feedback": evaluator_feedback
                })
        
        # Build prompt for LLM
        prompt = f"""Analyze this capstone viva examination and assess the learner's performance.

**Viva Metadata:**
- Questions asked: {questions_asked}
- Duration: {duration_minutes} minutes

**Question-Answer Pairs:**
{json.dumps(qa_pairs, indent=2)}

**Task:** For each Q&A pair, analyze:
1. **Is the answer satisfactory?** Look for:
   - Solution-oriented language ("I would...", "The fix is...", "should use...")
   - Correct technical reasoning
   - Evaluator positive feedback ("Good", "Correct", "Well reasoned")
   - Negative signals: "I don't know", vague answers, admission of weakness

2. **Extract topics** for weak and strong areas. Be descriptive:
   - Good: "error handling in tool nodes with conditional edges to fallback"
   - Bad: "error handling"

Dont invent Any answers on your own. Evaluate only on the actual Viva transcript.
If data is not available you dont need to create your own data.

**Output JSON only:**
Example output:
```json
{{
  "answers_satisfactory": 6,
  "weak_areas_identified": [
    "webhook signature verification for third-party integrations",
    "database transaction isolation levels and deadlock prevention"
  ],
  "strong_areas_identified": [
    "RESTful API design with proper resource modeling and verb selection",
    "authentication flow using JWT with refresh token rotation strategy",
    "database indexing strategy for complex queries with JOIN operations",
    "caching layer design using Redis with proper invalidation patterns"
  ]
}}
```"""
        
        response = llm_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": "You are a viva examiner assessor. Analyze answers objectively and extract descriptive topics. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        llm_result = json.loads(result_text)
        
        answers_satisfactory = llm_result.get("answers_satisfactory", 0)
        weak_areas = llm_result.get("weak_areas_identified", [])
        strong_areas = llm_result.get("strong_areas_identified", [])
        
        # Compute viva score
        viva_score = round((answers_satisfactory / questions_asked * 100)) if questions_asked > 0 else 0
        
        # Determine verdict
        agent_verdict = "pass" if viva_score >= 70 else "fail"
        
        result = {
            "capstone_id": data.get("capstone_id"),
            "viva_session_id": data.get("session_id"),
            "timeline_adherence": "on_time",
            "questions_asked": questions_asked,
            "duration_minutes": duration_minutes,
            "answers_satisfactory": answers_satisfactory,
            "weak_areas_identified": weak_areas,
            "strong_areas_identified": strong_areas,
            "viva_score": viva_score,
            "agent_verdict": agent_verdict
        }
        
        log.info(f"LLM computed viva summary: questions={questions_asked}, satisfactory={answers_satisfactory}, score={viva_score}, verdict={agent_verdict}")
        return json.dumps(result)
    
    except Exception as e:
        log.error("compute_viva_summary error: %s", e, exc_info=True)
        return json.dumps({"status": "error", "detail": str(e)})


@mcp.tool()
def mark_viva_processed(user_id: str, capstone_id: str, attempt_id: str, episode_id: str) -> str:
    """
    Mark all raw_viva_turns rows for a capstone attempt as processed.
    Call only after write_episode succeeds.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE raw_viva_turns
                SET    processed_at = now(), episode_id = %s
                WHERE  user_id = %s AND capstone_id = %s AND attempt_id = %s
                  AND  processed_at IS NULL
            """, (episode_id, user_id, capstone_id, attempt_id))
        conn.commit()
        log.info("raw_viva_turns marked processed: user=%s capstone=%s episode=%s",
                 user_id, capstone_id, episode_id)
        return "ok"
    except Exception as e:
        conn.rollback()
        log.error("mark_viva_processed error: %s", e)
        return f"error: {e}"
    finally:
        conn.close()


# =============================================================================
# LOGGING
# =============================================================================

@mcp.tool()
def log_extraction_error(user_id: str, session_id: str, episode_type: str, error_detail: str) -> str:
    log.error("EXTRACTION FAILED | user=%s session=%s type=%s | %s",
              user_id, session_id, episode_type, error_detail)
    return "logged"


# =============================================================================
# SEMANTIC PROFILE TOOLS
# =============================================================================

@mcp.tool()
def fetch_unprocessed_semantic_trigger(user_id: str) -> str:
    """
    Fetch the oldest unprocessed semantic rebuild trigger for a user.
    Returns {} if no unprocessed triggers exist.
    """
    log.info(f"Fetching unprocessed semantic trigger for user={user_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, user_id, trigger_reason, source_episode_id, triggered_at
                FROM   semantic_rebuild_trigger
                WHERE  user_id = %s
                  AND  processed_at IS NULL
                ORDER  BY triggered_at ASC
                LIMIT  1
            """, (user_id,))
            row = cur.fetchone()
            if not row:
                log.debug(f"No unprocessed semantic trigger found for user={user_id}")
                return json.dumps({})
            log.debug(f"SQL returned semantic trigger: {dict(row)}")
            result = dict(row)
            result["triggered_at"] = result["triggered_at"].isoformat()
            # Ensure source_episode_id is always a string (empty string if NULL)
            if result["source_episode_id"] is None:
                result["source_episode_id"] = ""
            log.info(f"Successfully fetched semantic trigger for user={user_id}: reason={result['trigger_reason']}")
            return json.dumps(result)
    finally:
        conn.close()


@mcp.tool()
def fetch_recent_episodes(user_id: str, since_timestamp: str, limit: int) -> str:
    """
    Fetch recent episodes for a user since a given timestamp.
    Used for semantic profile aggregation.
    If since_timestamp is empty, fetches all episodes (first profile build).
    Returns array of episodes with type and data fields.
    """
    log.info(f"Fetching recent episodes for user={user_id}, since={since_timestamp or 'all'}, limit={limit}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if since_timestamp:
                cur.execute("""
                    SELECT episode_id, type, timestamp, data
                    FROM   episodic_episodes
                    WHERE  user_id = %s
                      AND  timestamp > %s
                    ORDER  BY timestamp ASC
                    LIMIT  %s
                """, (user_id, since_timestamp, limit))
            else:
                cur.execute("""
                    SELECT episode_id, type, timestamp, data
                    FROM   episodic_episodes
                    WHERE  user_id = %s
                    ORDER  BY timestamp ASC
                    LIMIT  %s
                """, (user_id, limit))
            
            rows = cur.fetchall()
            log.debug(f"SQL returned {len(rows)} episodes")
            episodes = []
            for r in rows:
                episodes.append({
                    "episode_id": r["episode_id"],
                    "type": r["type"],
                    "timestamp": r["timestamp"].isoformat(),
                    "data": r["data"]
                })
            log.info(f"Successfully fetched {len(episodes)} episodes for user={user_id}")
            return json.dumps({"user_id": user_id, "episodes": episodes})
    finally:
        conn.close()


@mcp.tool()
def fetch_current_semantic_profile(user_id: str) -> str:
    """
    Fetch the current semantic profile for a user.
    Returns the profile JSONB directly, plus metadata (version, is_new, last_updated).
    For new users, returns empty template with is_new=true.
    """
    log.info(f"Fetching current semantic profile for user={user_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id, version, profile, trigger, created_at
                FROM   semantic_profile_versions
                WHERE  user_id = %s
                  AND  is_current = true
                LIMIT  1
            """, (user_id,))
            row = cur.fetchone()
            
            if not row:
                # Return empty template for new users
                empty_profile = {
                    "user_id": user_id,
                    "version": 0,
                    "last_updated": None,
                    "last_updated_trigger": None,
                    "identity": {
                        "declared_interests": [],
                        "primary_focus": None,
                        "learning_goal": None
                    },
                    "skill_mastery_ref": user_id,
                    "skill_note": "Skill levels live in user_skill_mastery — join on user_id at query time",
                    "performance_profile": {
                        "avg_exercise_score": None,
                        "avg_attempts_per_exercise": None,
                        "timeline_adherence": None,
                        "engagement_pattern": None,
                        "avg_session_duration_minutes": None
                    },
                    "known_struggles": [],
                    "known_strengths": [],
                    "capstone_history": [],
                    "mentor_context": {
                        "preferred_explanation_style": "adaptive",
                        "common_question_themes": [],
                        "last_interaction_summary": None,
                        "motivation_signals": "new learner"
                    }
                }
                log.info(f"No existing semantic profile found for user={user_id}, returning empty template")
                return json.dumps({
                    "profile": empty_profile,
                    "version": 0,
                    "is_new": True,
                    "last_updated": None
                })
            
            # Extract profile JSONB and metadata
            profile_data = row["profile"]
            last_updated = profile_data.get("last_updated") if profile_data else None
            
            log.info(f"Successfully fetched semantic profile for user={user_id}, version={row['version']}")
            return json.dumps({
                "profile": profile_data,
                "version": row["version"],
                "is_new": False,
                "last_updated": last_updated
            })
    finally:
        conn.close()


def _analyze_learning_signals_with_llm(
    session_hours: list,
    session_weekdays: list,
    topic_negative_signals: dict,
    topic_positive_signals: dict,
    total_episodes: int,
    mentor_exchanges: list,
    question_type_counts: dict
) -> dict:
    """
    Helper: Use LLM to analyze all learning signals and infer:
    - engagement_pattern
    - known_struggles (with severity)
    - known_strengths
    - motivation_signals
    - timeline_adherence
    
    Replaces ALL hardcoded business logic with LLM-based inference.
    """
    prompt = f"""Analyze this learner's behavioral signals and create a psychological profile.

**Session Timing Data:**
- Session hours (24hr): {session_hours[:50] if len(session_hours) > 50 else session_hours}
- Session weekdays (0=Mon, 6=Sun): {session_weekdays[:50] if len(session_weekdays) > 50 else session_weekdays}
- Total episodes: {total_episodes}

**Topic Performance Signals:**
Negative signals (struggles, low scores, many attempts, overconfidence):
{json.dumps(dict(list(topic_negative_signals.items())[:10]), indent=2)}

Positive signals (high scores, quick mastery, underconfidence showing hidden strength):
{json.dumps(dict(list(topic_positive_signals.items())[:10]), indent=2)}

**Mentor Interaction Context:**
- Question types: {json.dumps(question_type_counts)}
- Recent exchanges: {json.dumps(mentor_exchanges[:3], indent=2) if mentor_exchanges else "none"}

**Your Task:**
Analyze the data and output a JSON object with:

1. **engagement_pattern**: Infer from session_hours and session_weekdays. Options: "morning_learner", "evening_learner", "weekend_learner", "late_night_learner", "varied", "sporadic"

2. **known_struggles**: Array of topics where learner struggles. For each:
   - "topic": topic name
   - "severity": "high", "moderate", or "mild" (consider frequency, recency, type of signal)
   - "last_seen": most recent timestamp string
   Sort by severity then recency.

3. **known_strengths**: Array of topics where learner excels:
   - "topic": topic name  
   - "last_evidenced": most recent timestamp string
   Sort by recency.

4. **motivation_signals**: String describing engagement quality. Examples: "consistent daily engagement", "sporadic bursts", "declining activity", "weekend warrior", "new learner"

5. **timeline_adherence**: "excellent", "good", "falling_behind", "inconsistent", or "unknown"

**CRITICAL:** Base severity on:
- Signal frequency (more = worse, but context matters)
- Recency (recent = more relevant)
- Signal type (viva_weakness > low exercise score)
- Improvement trajectory (if improving, lower severity)

Do NOT use arbitrary thresholds. Use your judgment.

Output ONLY valid JSON. No markdown, no explanation."""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an educational psychologist analyzing learner data. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        llm_result = json.loads(result_text)
        
        log.info(f"LLM analyzed learning signals: pattern={llm_result.get('engagement_pattern')}, struggles={len(llm_result.get('known_struggles', []))}")
        return llm_result
    
    except Exception as e:
        log.error(f"LLM analysis error: {e}")
        # Fallback to minimal safe defaults
        return {
            "engagement_pattern": "varied",
            "known_struggles": [],
            "known_strengths": [],
            "motivation_signals": "unknown",
            "timeline_adherence": "unknown"
        }


@mcp.tool()
def compute_semantic_profile_update(user_id: str) -> str:
    """
    Server-side computation of semantic profile updates from episodes.
    Fetches current profile and all episodes, performs deterministic aggregations,
    and uses LLM to infer semantic fields (learning_goal, explanation_style, question_themes).
    
    Returns complete enhanced profile ready for writing. No agent reasoning required.
    """
    log.info(f"Computing semantic profile update for user={user_id}")
    conn = get_conn()
    try:
        # Step 1: Fetch current profile
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT user_id, version, profile, trigger, created_at
                FROM   semantic_profile_versions
                WHERE  user_id = %s
                  AND  is_current = true
                LIMIT  1
            """, (user_id,))
            profile_row = cur.fetchone()
            
            if not profile_row:
                # Create empty template for new users
                current_profile = {
                    "user_id": user_id,
                    "version": 0,
                    "last_updated": None,
                    "last_updated_trigger": None,
                    "identity": {
                        "declared_interests": [],
                        "primary_focus": None,
                        "learning_goal": None
                    },
                    "skill_mastery_ref": user_id,
                    "skill_note": "Skill levels live in user_skill_mastery — join on user_id at query time",
                    "performance_profile": {
                        "avg_exercise_score": None,
                        "avg_attempts_per_exercise": None,
                        "timeline_adherence": None,
                        "engagement_pattern": None,
                        "avg_session_duration_minutes": None
                    },
                    "known_struggles": [],
                    "known_strengths": [],
                    "capstone_history": [],
                    "mentor_context": {
                        "preferred_explanation_style": "adaptive",
                        "common_question_themes": [],
                        "last_interaction_summary": None,
                        "motivation_signals": "new learner"
                    }
                }
            else:
                current_profile = profile_row["profile"]
        
        # Step 2: Fetch all episodes
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT episode_id, type, timestamp, data
                FROM   episodic_episodes
                WHERE  user_id = %s
                ORDER  BY timestamp ASC
                LIMIT  1000
            """, (user_id,))
            
            rows = cur.fetchall()
            log.info(f"Fetched {len(rows)} episodes for user={user_id}")
            episodes = []
            for r in rows:
                episodes.append({
                    "episode_id": r["episode_id"],
                    "type": r["type"],
                    "timestamp": r["timestamp"].isoformat(),
                    "data": r["data"]
                })
        
        # Initialize counters and accumulators
        exercise_scores = []
        exercise_attempts = []
        session_durations = []
        session_hours = []  # Hour of day for each session
        session_weekdays = []  # Day of week (0=Mon, 6=Sun)
        
        topic_negative_signals = {}  # topic -> list of negative signal objects
        topic_positive_signals = {}  # topic -> list of positive signal objects
        
        declared_interests = set(current_profile.get("identity", {}).get("declared_interests", []))
        pathway_activity = {}  # interest -> episode count
        
        capstone_data = {}  # capstone_id -> aggregated data
        
        mentor_exchanges = []
        question_type_counts = {"clarification": 0, "stuck": 0, "curiosity": 0}
        
        # Process each episode
        for ep in episodes:
            ep_type = ep["type"]
            data = ep["data"]
            timestamp = ep["timestamp"]
            
            # SELF_ASSESSMENT
            if ep_type == "SELF_ASSESSMENT":
                interests = data.get("declared_interests", [])
                declared_interests.update(interests)
            
            # PRE_ASSESSMENT - calibration signals
            elif ep_type == "PRE_ASSESSMENT":
                calibration_delta = data.get("calibration_delta", {})
                for domain, topics in calibration_delta.items():
                    for topic, delta in topics.items():
                        # Collect signals for LLM (no hardcoded meaning)
                        if topic not in topic_negative_signals:
                            topic_negative_signals[topic] = []
                        if topic not in topic_positive_signals:
                            topic_positive_signals[topic] = []
                        
                        if delta == "overconfident":
                            topic_negative_signals[topic].append({
                                "delta": delta,
                                "timestamp": timestamp,
                                "type": "calibration"
                            })
                        elif delta == "underconfident":
                            topic_positive_signals[topic].append({
                                "delta": delta,
                                "timestamp": timestamp,
                                "type": "calibration"
                            })
            
            # COURSE_ACTIVITY
            elif ep_type == "COURSE_ACTIVITY":
                # Track pathway activity
                course_id = data.get("course_id")
                if course_id:
                    for interest in declared_interests:
                        pathway_activity[interest] = pathway_activity.get(interest, 0) + 1
                
                # Exercise performance
                exercises = data.get("ide_exercises", [])
                for ex in exercises:
                    final_score = ex.get("final_score")
                    attempts = ex.get("attempts")
                    topic = ex.get("topic_id")
                    
                    if final_score is not None:
                        exercise_scores.append(final_score)
                        
                        # Collect signals for LLM analysis (simple categorization, LLM determines severity)
                        if topic:
                            # Negative signal: low score or many attempts
                            if final_score < 70 or (attempts and attempts > 2):
                                if topic not in topic_negative_signals:
                                    topic_negative_signals[topic] = []
                                topic_negative_signals[topic].append({
                                    "score": final_score,
                                    "attempts": attempts,
                                    "timestamp": timestamp,
                                    "type": "exercise"
                                })
                            
                            # Positive signal: high score with few attempts
                            elif final_score > 80 and (not attempts or attempts <= 2):
                                if topic not in topic_positive_signals:
                                    topic_positive_signals[topic] = []  
                                topic_positive_signals[topic].append({
                                    "score": final_score,
                                    "attempts": attempts,
                                    "timestamp": timestamp,
                                    "type": "exercise"
                                })
                    
                    if attempts is not None:
                        exercise_attempts.append(attempts)
                
                # Session metrics
                duration = data.get("duration_minutes")
                if duration:
                    session_durations.append(duration)
                
                # Parse timestamp for engagement pattern
                try:
                    from datetime import datetime as dt
                    ts = dt.fromisoformat(timestamp.replace('Z', '+00:00'))
                    session_hours.append(ts.hour)
                    session_weekdays.append(ts.weekday())
                except:
                    pass
            
            # MENTOR_CHAT
            elif ep_type == "MENTOR_CHAT":
                duration = data.get("duration_minutes")
                if duration:
                    session_durations.append(duration)
                
                # Unresolved questions = struggles
                unresolved = data.get("unresolved_questions", [])
                for q in unresolved:
                    # Extract topic from question (simplified - agent can do better)
                    # For now, just mark as generic struggle
                    pass
                
                # Question types
                q_types = data.get("question_types", {})
                question_type_counts["clarification"] += q_types.get("clarification", 0)
                question_type_counts["stuck"] += q_types.get("stuck", 0)
                question_type_counts["curiosity"] += q_types.get("curiosity", 0)
                
                # Store exchanges for agent processing
                exchanges = data.get("exchanges_summary", [])
                if exchanges:
                    mentor_exchanges.extend(exchanges[:3])  # Keep recent ones
                
                # Parse timestamp
                try:
                    from datetime import datetime as dt
                    ts = dt.fromisoformat(timestamp.replace('Z', '+00:00'))
                    session_hours.append(ts.hour)
                    session_weekdays.append(ts.weekday())
                except:
                    pass
            
            # CAPSTONE episodes
            elif ep_type == "CAPSTONE_CODE_REVIEW":
                capstone_id = data.get("capstone_id")
                if capstone_id:
                    if capstone_id not in capstone_data:
                        capstone_data[capstone_id] = {
                            "capstone_id": capstone_id,
                            "status": "in_progress",
                            "code_review_verdict": None,
                            "tech_verdicts": {},
                            "test_verdict": None,
                            "viva_verdict": None,
                            "viva_score": None
                        }
                    # Extract verdict from review output
                    review_verdict = data.get("overall_verdict", data.get("code_review_verdict", "needs_revision"))
                    capstone_data[capstone_id]["code_review_verdict"] = review_verdict
                    
                    # Extract tech verdicts from tech_evaluations (LLM output from compute_code_review_summary)
                    tech_evaluations = data.get("tech_evaluations", [])
                    for tech_eval in tech_evaluations:
                        tech_name = tech_eval.get("technology", "")
                        # Synthesize verdict from evaluations
                        code_quality = tech_eval.get("code_quality", "")
                        understanding = tech_eval.get("understanding", "")
                        # Simple heuristic: if mentions "confusion", "missed", "weak" = needs_work, else ok
                        combined = (code_quality + " " + understanding).lower()
                        if any(word in combined for word in ["confusion", "missed", "weak", "poor", "inadequate"]):
                            verdict = "needs_work"
                        elif any(word in combined for word in ["strong", "solid", "good", "excellent", "demonstrates"]):
                            verdict = "good"
                        else:
                            verdict = "adequate"
                        
                        if tech_name:
                            # Ensure strings are safe for JSON
                            capstone_data[capstone_id]["tech_verdicts"][tech_name] = {
                                "verdict": verdict,
                                "understanding": str(understanding[:100]).replace('"', "'"),
                                "code_quality": str(code_quality[:100]).replace('"', "'")
                            }
            
            elif ep_type == "CAPSTONE_TEST_RUN":
                capstone_id = data.get("capstone_id")
                if capstone_id:
                    if capstone_id not in capstone_data:
                        capstone_data[capstone_id] = {
                            "capstone_id": capstone_id,
                            "status": "in_progress",
                            "code_review_verdict": None,
                            "tech_verdicts": {},
                            "test_verdict": None,
                            "viva_verdict": None,
                            "viva_score": None
                        }
                    overall_passed = data.get("overall_passed", False)
                    capstone_data[capstone_id]["test_verdict"] = "pass" if overall_passed else "fail"
            
            elif ep_type == "CAPSTONE_VIVA":
                capstone_id = data.get("capstone_id")
                if capstone_id:
                    if capstone_id not in capstone_data:
                        capstone_data[capstone_id] = {
                            "capstone_id": capstone_id,
                            "status": "in_progress",
                            "code_review_verdict": None,
                            "tech_verdicts": {},
                            "test_verdict": None,
                            "viva_verdict": None,
                            "viva_score": None
                        }
                    viva_score = data.get("viva_score")
                    agent_verdict = data.get("agent_verdict", "fail")
                    
                    capstone_data[capstone_id]["viva_verdict"] = agent_verdict
                    capstone_data[capstone_id]["viva_score"] = viva_score
                    
                    # Update status based on viva pass/fail
                    if agent_verdict == "pass":
                        capstone_data[capstone_id]["status"] = "passed"
                    else:
                        capstone_data[capstone_id]["status"] = "failed"
                    
                    # Extract weak areas as struggles (viva-level insight)
                    weak_areas = data.get("weak_areas_identified", [])
                    for area in weak_areas:
                        if area not in topic_negative_signals:
                            topic_negative_signals[area] = []
                        topic_negative_signals[area].append({
                            "timestamp": timestamp,
                            "type": "viva_weakness",
                            "source": "capstone_viva"
                        })
                    
                    # Extract strong areas as strengths
                    strong_areas = data.get("strong_areas_identified", [])
                    for area in strong_areas:
                        if area not in topic_positive_signals:
                            topic_positive_signals[area] = []
                        topic_positive_signals[area].append({
                            "timestamp": timestamp,
                            "type": "viva_strength",
                            "source": "capstone_viva"
                        })
        
        # Compute aggregated metrics (no hardcoded inference)
        from datetime import datetime as dt, timezone
        
        # Performance profile
        avg_exercise_score = sum(exercise_scores) / len(exercise_scores) if exercise_scores else None
        avg_attempts = sum(exercise_attempts) / len(exercise_attempts) if exercise_attempts else None
        avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else None
        
        # Extract recent activity (last 3 sessions)
        recent_sessions = []
        session_episodes = [ep for ep in episodes if ep["type"] in ["COURSE_ACTIVITY", "MENTOR_CHAT"]]
        for ep in session_episodes[-3:]:
            # Safely extract summary, avoiding JSON serialization issues
            summary = "session"
            if ep["type"] == "COURSE_ACTIVITY":
                summary = ep["data"].get("course_id", "course_session")
            elif ep["type"] == "MENTOR_CHAT":
                summary = f"mentor_chat_{ep['data'].get('session_id', 'unknown')}"
            
            recent_sessions.append({
                "type": ep["type"],
                "timestamp": ep["timestamp"],
                "summary": str(summary)  # Ensure string type
            })
        
        # Query student_pathways for learning trajectory
        learning_trajectory = None
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT pathway_id, interest, status, 
                           courses_in_progress, courses_completed, courses_remaining,
                           capstone_id
                    FROM student_pathways
                    WHERE user_id = %s AND status = 'active'
                    LIMIT 1
                """, (user_id,))
                pathway_row = cur.fetchone()
                if pathway_row:
                    # Calculate progress metrics
                    completed_count = len(pathway_row["courses_completed"]) if pathway_row["courses_completed"] else 0
                    in_progress_count = len(pathway_row["courses_in_progress"]) if pathway_row["courses_in_progress"] else 0
                    remaining_count = len(pathway_row["courses_remaining"]) if pathway_row["courses_remaining"] else 0
                    total_courses = completed_count + in_progress_count + remaining_count
                    progress_percent = round((completed_count / total_courses * 100)) if total_courses > 0 else 0
                    
                    learning_trajectory = {
                        "pathway_id": pathway_row["pathway_id"],
                        "interest": pathway_row["interest"],
                        "status": pathway_row["status"],
                        "current_courses": pathway_row["courses_in_progress"],
                        "completed_courses": pathway_row["courses_completed"],
                        "remaining_courses": pathway_row["courses_remaining"],
                        "completed_count": completed_count,
                        "total_courses": total_courses,
                        "progress_percent": progress_percent,
                        "capstone_id": pathway_row["capstone_id"]
                    }
        except Exception as e:
            log.warning(f"Could not fetch learning_trajectory: {e}")
        
        # Primary focus (most active interest)
        primary_focus = None
        if pathway_activity:
            primary_focus = max(pathway_activity, key=pathway_activity.get)
        elif declared_interests:
            primary_focus = list(declared_interests)[0]
        
        # LLM INFERENCE: Analyze all signals to determine struggles, strengths, patterns
        llm_analysis = _analyze_learning_signals_with_llm(
            session_hours=session_hours,
            session_weekdays=session_weekdays,
            topic_negative_signals=topic_negative_signals,
            topic_positive_signals=topic_positive_signals,
            total_episodes=len(episodes),
            mentor_exchanges=mentor_exchanges,
            question_type_counts=question_type_counts
        )
        
        # Build updated profile (using LLM inferences, no hardcoded rules)
        updated_profile = {
            "user_id": current_profile.get("user_id"),
            "version": current_profile.get("version", 0),  # Will be incremented by write tool
            "last_updated": None,  # Will be set by write tool
            "last_updated_trigger": None,  # Will be set by write tool
            
            "identity": {
                "declared_interests": sorted(list(declared_interests)),
                "primary_focus": primary_focus,
                "learning_goal": current_profile.get("identity", {}).get("learning_goal")  # Preserved, set by agent
            },
            
            "skill_mastery_ref": current_profile.get("user_id"),
            "skill_note": "Skill levels live in user_skill_mastery — join on user_id at query time",
            
            "performance_profile": {
                "avg_exercise_score": round(avg_exercise_score, 1) if avg_exercise_score else None,
                "avg_attempts_per_exercise": round(avg_attempts, 1) if avg_attempts else None,
                "timeline_adherence": llm_analysis.get("timeline_adherence", "unknown"),
                "engagement_pattern": llm_analysis.get("engagement_pattern", "varied"),
                "avg_session_duration_minutes": round(avg_session_duration, 1) if avg_session_duration else None
            },
            
            "known_struggles": llm_analysis.get("known_struggles", []),
            "known_strengths": llm_analysis.get("known_strengths", []),
            "capstone_history": list(capstone_data.values()),
            
            "recent_activity": recent_sessions,
            "learning_trajectory": learning_trajectory,
            
            "mentor_context": {
                "preferred_explanation_style": current_profile.get("mentor_context", {}).get("preferred_explanation_style", "adaptive"),
                "common_question_themes": current_profile.get("mentor_context", {}).get("common_question_themes", []),
                "last_interaction_summary": mentor_exchanges[0] if mentor_exchanges else None,
                "motivation_signals": llm_analysis.get("motivation_signals", "new learner")
            },
            
            # Additional data for agent reasoning
            "_agent_context": {
                "question_type_counts": question_type_counts,
                "mentor_exchanges": mentor_exchanges,
                "total_episodes": len(episodes)
            }
        }
        
        # LLM REASONING: Infer semantic fields
        llm_inferred = _infer_semantic_fields_with_llm(updated_profile, episodes)
        
        # Apply LLM inferences to profile
        updated_profile["identity"]["learning_goal"] = llm_inferred["learning_goal"]
        updated_profile["mentor_context"]["preferred_explanation_style"] = llm_inferred["preferred_explanation_style"]
        updated_profile["mentor_context"]["common_question_themes"] = llm_inferred["common_question_themes"]
        
        # Remove _agent_context before returning (no longer needed)
        updated_profile.pop("_agent_context", None)
        
        # Validate JSON serialization before returning
        try:
            profile_json = json.dumps(updated_profile)
            # Test parse to catch any issues
            json.loads(profile_json)
        except Exception as json_err:
            log.error(f"JSON serialization error in compute_semantic_profile_update: {json_err}")
            log.error(f"Problematic profile keys: {list(updated_profile.keys())}")
            # Return error
            return json.dumps({"status": "error", "detail": f"JSON serialization failed: {str(json_err)}"})
        
        log.info(f"Successfully computed semantic profile update: {len(episodes)} episodes, "
                 f"{len(llm_analysis.get('known_struggles', []))} struggles, "
                 f"{len(llm_analysis.get('known_strengths', []))} strengths")
        return profile_json
    
    except Exception as e:
        log.error(f"compute_semantic_profile_update error for user={user_id}: {e}")
        return json.dumps({"status": "error", "detail": str(e)})
    finally:
        conn.close()


@mcp.tool()
def write_semantic_rebuild_trigger(user_id: str, trigger_reason: str, source_episode_id: str) -> str:
    """
    Write a semantic rebuild trigger.
    Idempotent: checks for duplicate trigger within 10-minute window.
    Returns trigger_id and status.
    """
    log.info(f"Writing semantic rebuild trigger for user={user_id}, reason={trigger_reason}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Check for duplicate trigger
            cur.execute("""
                SELECT id
                FROM   semantic_rebuild_trigger
                WHERE  user_id = %s
                  AND  trigger_reason = %s
                  AND  triggered_at > now() - interval '10 minutes'
                LIMIT  1
            """, (user_id, trigger_reason))
            existing = cur.fetchone()
            
            if existing:
                log.debug(f"Duplicate semantic trigger blocked for user={user_id}, reason={trigger_reason}")
                return json.dumps({
                    "trigger_id": existing["id"],
                    "status": "ok",
                    "note": "duplicate_blocked"
                })
            
            trigger_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO semantic_rebuild_trigger
                    (id, user_id, trigger_reason, source_episode_id, triggered_at)
                VALUES (%s, %s, %s, %s, now())
            """, (trigger_id, user_id, trigger_reason, source_episode_id))
        
        conn.commit()
        log.info(f"Successfully wrote semantic rebuild trigger: trigger_id={trigger_id}, user={user_id}")
        return json.dumps({"trigger_id": trigger_id, "status": "ok"})
    except Exception as e:
        conn.rollback()
        log.error("write_semantic_rebuild_trigger error: %s", e)
        return json.dumps({"status": "error", "detail": str(e)})
    finally:
        conn.close()


@mcp.tool()
def write_semantic_profile_version(user_id: str, profile_json: str, trigger_reason: str, 
                                     source_episode_id: str) -> str:
    """
    Write a new semantic profile version to semantic_profile_versions table.
    Atomically updates is_current flags and increments version number.
    Returns new version number and status.
    source_episode_id is for audit trail (pass empty string if not available).
    """
    log.info(f"write_semantic_profile_version called: user={user_id}, trigger={trigger_reason}")
    conn = get_conn()
    try:
        profile = json.loads(profile_json)
        log.info(f"Parsed profile JSON, keys: {list(profile.keys())}")
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get current max version
            cur.execute("""
                SELECT COALESCE(MAX(version), 0) as max_version
                FROM   semantic_profile_versions
                WHERE  user_id = %s
            """, (user_id,))
            row = cur.fetchone()
            new_version = row["max_version"] + 1
            log.info(f"Determined new version: {new_version}")
            
            # Update profile with new version and timestamp
            profile["version"] = new_version
            profile["last_updated"] = datetime.now(timezone.utc).isoformat()
            profile["last_updated_trigger"] = trigger_reason
            
            # Unset is_current on all existing profiles
            cur.execute("""
                UPDATE semantic_profile_versions
                SET    is_current = false
                WHERE  user_id = %s
            """, (user_id,))
            updated_count = cur.rowcount
            log.info(f"Unmarked {updated_count} existing profiles as not current")
            
            # Insert new version
            log.info(f"Inserting new version {new_version} into semantic_profile_versions")
            cur.execute("""
                INSERT INTO semantic_profile_versions
                    (user_id, version, profile, trigger, created_at, is_current)
                VALUES (%s, %s, %s, %s, now(), true)
            """, (user_id, new_version, json.dumps(profile), trigger_reason))
            insert_count = cur.rowcount
            log.info(f"INSERT completed, rowcount={insert_count}")
        
        conn.commit()
        log.info(f"✓ COMMITTED semantic profile v{new_version} for user {user_id} (trigger: {trigger_reason})")
        
        # Verify the insert
        with conn.cursor() as verify_cur:
            verify_cur.execute("""
                SELECT version FROM semantic_profile_versions 
                WHERE user_id = %s AND version = %s
            """, (user_id, new_version))
            verify_row = verify_cur.fetchone()
            if verify_row:
                log.info(f"✓ VERIFIED: Version {new_version} exists in database")
            else:
                log.error(f"✗ VERIFICATION FAILED: Version {new_version} NOT FOUND after commit!")
        
        return json.dumps({"version": new_version, "status": "ok"})
    
    except Exception as e:
        conn.rollback()
        log.error(f"✗ write_semantic_profile_version ERROR for user={user_id}: {e}", exc_info=True)
        return json.dumps({"status": "error", "detail": str(e)})
    finally:
        conn.close()
        log.info(f"Connection closed for user={user_id}")


@mcp.tool()
def mark_semantic_rebuild_processed(trigger_id: str, new_version: int) -> str:
    """
    Mark a semantic rebuild trigger as processed.
    Call after successfully writing new profile version.
    Returns 'ok' or error.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE semantic_rebuild_trigger
                SET    processed_at = now()
                WHERE  id = %s
                  AND  processed_at IS NULL
            """, (trigger_id,))
        conn.commit()
        log.info("Semantic trigger %s marked processed (new version: %s)", 
                 trigger_id, new_version)
        return "ok"
    except Exception as e:
        conn.rollback()
        log.error("mark_semantic_rebuild_processed error: %s", e)
        return f"error: {e}"
    finally:
        conn.close()


# =============================================================================
# SEMANTIC QUERY TOOLS
# =============================================================================

@mcp.tool()
def query_knowledge_gaps(user_id: str, domain: str) -> str:
    """
    Query known struggles from semantic profile filtered by domain.
    Returns topics with struggle signals, sorted by severity and recency.
    Useful for Pathway agent (avoid topics) and Mentor chatbot (proactive help).
    """
    log.info(f"Querying knowledge gaps for user={user_id}, domain={domain}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT profile
                FROM   semantic_profile_versions
                WHERE  user_id = %s
                  AND  is_current = true
                LIMIT  1
            """, (user_id,))
            row = cur.fetchone()
            
            if not row:
                return json.dumps({"gaps": []})
            
            profile = row["profile"]
            known_struggles = profile.get("known_struggles", [])
            
            # Filter by domain if needed (would need topic-to-domain mapping from FalkorDB)
            # For now, return all struggles
            gaps = [
                {
                    "topic": s["topic"],
                    "severity": s["severity"],
                    "last_seen": s["last_seen"]
                }
                for s in known_struggles
                if s.get("severity") in ["moderate", "high"]  # Filter out mild struggles
            ]
            
            log.info(f"Successfully queried knowledge gaps for user={user_id}: {len(gaps)} gaps found")
            return json.dumps({"user_id": user_id, "domain": domain, "gaps": gaps})
    finally:
        conn.close()


@mcp.tool()
def query_learning_velocity(user_id: str, time_period_days: int) -> str:
    """
    Compute learning velocity: concepts mastered per time period.
    Returns count of topics added to known_strengths in the period and trend direction.
    """
    log.info(f"Querying learning velocity for user={user_id}, time_period={time_period_days} days")
    conn = get_conn()
    try:
        from datetime import datetime as dt, timezone, timedelta
        
        cutoff = dt.now(timezone.utc) - timedelta(days=time_period_days)
        cutoff_str = cutoff.isoformat()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT profile
                FROM   semantic_profile_versions
                WHERE  user_id = %s
                  AND  is_current = true
                LIMIT  1
            """, (user_id,))
            row = cur.fetchone()
            
            if not row:
                return json.dumps({"velocity": 0, "trend": "no_data"})
            
            profile = row["profile"]
            known_strengths = profile.get("known_strengths", [])
            
            # Count strengths evidenced within time period
            recent_count = 0
            for strength in known_strengths:
                last_evidenced = strength.get("last_evidenced")
                if last_evidenced and last_evidenced >= cutoff_str:
                    recent_count += 1
            
            velocity = recent_count / max(1, time_period_days / 7)  # Per week
            
            # Simple trend detection (would be better with historical versions)
            trend = "steady"
            if velocity >= 2:
                trend = "accelerating"
            elif velocity < 0.5:
                trend = "slowing"
            
            log.info(f"Successfully computed learning velocity for user={user_id}: {recent_count} concepts, velocity={velocity:.2f}/week")
            return json.dumps({
                "user_id": user_id,
                "time_period_days": time_period_days,
                "concepts_mastered": recent_count,
                "velocity_per_week": round(velocity, 2),
                "trend": trend
            })
    finally:
        conn.close()


@mcp.tool()
def query_recommended_topics(user_id: str, domain: str) -> str:
    """
    Suggest next topics based on:
    - Current pathway position (from student_pathways)
    - Known strengths (completed topics)
    - Known struggles (avoid or reinforce)
    - Skill mastery levels
    
    Returns array of recommended topic IDs with reasoning.
    """
    log.info(f"Querying recommended topics for user={user_id}, domain={domain}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get semantic profile
            cur.execute("""
                SELECT profile
                FROM   semantic_profile_versions
                WHERE  user_id = %s
                  AND  is_current = true
                LIMIT  1
            """, (user_id,))
            profile_row = cur.fetchone()
            
            # Get active pathway
            cur.execute("""
                SELECT courses_remaining, courses_completed
                FROM   student_pathways
                WHERE  user_id = %s
                  AND  status = 'active'
                  AND  interest = %s
                LIMIT  1
            """, (user_id, domain))
            pathway_row = cur.fetchone()
            
            if not profile_row or not pathway_row:
                return json.dumps({"recommendations": []})
            
            profile = profile_row["profile"]
            courses_remaining = pathway_row["courses_remaining"]
            courses_completed = pathway_row["courses_completed"]
            
            known_strengths_topics = {s["topic"] for s in profile.get("known_strengths", [])}
            known_struggles_topics = {s["topic"] for s in profile.get("known_struggles", [])}
            
            # Simple recommendation: next course in pathway
            recommendations = []
            if courses_remaining:
                next_course = courses_remaining[0]
                recommendations.append({
                    "item_type": "course",
                    "item_id": next_course,
                    "reasoning": "Next in your learning pathway",
                    "priority": "high"
                })
            
            # Suggest reinforcement for struggles
            for topic_data in profile.get("known_struggles", [])[:2]:  # Top 2 struggles
                if topic_data.get("severity") == "high":
                    recommendations.append({
                        "item_type": "topic_review",
                        "item_id": topic_data["topic"],
                        "reasoning": "Reinforce struggling topic",
                        "priority": "medium"
                    })
            
            log.info(f"Successfully queried recommended topics for user={user_id}: {len(recommendations)} recommendations")
            return json.dumps({
                "user_id": user_id,
                "domain": domain,
                "recommendations": recommendations
            })
    finally:
        conn.close()


@mcp.tool()
def get_mentor_personalization(user_id: str) -> str:
    """
    Get mentor_context section from semantic profile.
    Used by Mentor chatbot to adapt response style and context.
    Returns preferred_explanation_style, common_question_themes, 
    last_interaction_summary, motivation_signals.
    """
    log.info(f"Getting mentor personalization for user={user_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT profile
                FROM   semantic_profile_versions
                WHERE  user_id = %s
                  AND  is_current = true
                LIMIT  1
            """, (user_id,))
            row = cur.fetchone()
            
            if not row:
                # Return defaults for new users
                log.debug(f"No semantic profile found for user={user_id}, returning defaults")
                return json.dumps({
                    "user_id": user_id,
                    "preferred_explanation_style": "adaptive",
                    "common_question_themes": [],
                    "last_interaction_summary": None,
                    "motivation_signals": "new learner"
                })
            
            profile = row["profile"]
            mentor_context = profile.get("mentor_context", {})
            
            log.info(f"Successfully retrieved mentor personalization for user={user_id}")
            return json.dumps({
                "user_id": user_id,
                "preferred_explanation_style": mentor_context.get("preferred_explanation_style", "adaptive"),
                "common_question_themes": mentor_context.get("common_question_themes", []),
                "last_interaction_summary": mentor_context.get("last_interaction_summary"),
                "motivation_signals": mentor_context.get("motivation_signals", "new learner")
            })
    finally:
        conn.close()


# =============================================================================
# MENTOR CHATBOT TOOLS — Runtime conversation support
# =============================================================================

@mcp.tool()
def get_semantic_profile_slice(user_id: str, course_id: str = None) -> str:
    """
    Get learner's long-term profile for mentor personalization.
    Returns mentor_context, performance_profile, known_struggles, known_strengths.
    Call at START of each new mentor session.
    """
    log.info(f"Getting semantic profile slice for user={user_id}, course={course_id or 'all'}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT profile FROM semantic_profile_versions
                WHERE user_id = %s AND is_current = true
                LIMIT 1
            """, (user_id,))
            row = cur.fetchone()
            
            if not row:
                log.debug(f"No semantic profile found for user={user_id}, returning empty defaults")
                return json.dumps({
                    "user_id": user_id,
                    "mentor_context": {"preferred_explanation_style": "adaptive"},
                    "performance_profile": {},
                    "known_struggles": [],
                    "known_strengths": []
                })
            
            profile = row["profile"]
            struggles = profile.get("known_struggles", [])
            strengths = profile.get("known_strengths", [])
            
            # Filter by course if provided
            if course_id:
                struggles = [s for s in struggles if s.get("course_id") == course_id]
                strengths = [s for s in strengths if s.get("course_id") == course_id]
            
            log.info(f"Successfully retrieved semantic profile slice for user={user_id}: {len(struggles)} struggles, {len(strengths)} strengths")
            return json.dumps({
                "user_id": user_id,
                "mentor_context": profile.get("mentor_context", {}),
                "performance_profile": profile.get("performance_profile", {}),
                "known_struggles": struggles,
                "known_strengths": strengths
            })
    finally:
        conn.close()


@mcp.tool()
def get_skill_mastery(user_id: str, course_id: str = None) -> str:
    """
    Get learner's current skill mastery levels with time decay.
    Returns effective_score (0.0-1.0) and confidence for each skill.
    Call at START of each mentor session.
    """
    log.info(f"Getting skill mastery for user={user_id}, course={course_id or 'all'}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT profile FROM semantic_profile_versions
                WHERE user_id = %s AND is_current = true
                LIMIT 1
            """, (user_id,))
            row = cur.fetchone()
            
            if not row:
                return json.dumps({"user_id": user_id, "skills": []})
            
            profile = row["profile"]
            skills = profile.get("skill_mastery", [])
            
            # Filter by course if provided
            if course_id:
                skills = [s for s in skills if s.get("course_id") == course_id]
            
            log.info(f"Successfully retrieved skill mastery for user={user_id}: {len(skills)} skills")
            return json.dumps({
                "user_id": user_id,
                "course_id": course_id,
                "skills": skills
            })
    finally:
        conn.close()


@mcp.tool()
def get_lesson_content(lesson_id: str) -> str:
    """
    Fetch full lesson content by ID.
    Call at START of mentor session so you can reference lesson material.
    """
    log.info(f"Getting lesson content for lesson_id={lesson_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM lessons WHERE lesson_id = %s
            """, (lesson_id,))
            lesson = cur.fetchone()
            
            if not lesson:
                log.debug(f"Lesson {lesson_id} not found")
                return json.dumps({
                    "success": False,
                    "error": f"Lesson {lesson_id} not found"
                })
            
            log.info(f"Successfully retrieved lesson content for lesson_id={lesson_id}")
            return json.dumps({
                "success": True,
                "lesson_id": lesson["lesson_id"],
                "title": lesson.get("title"),
                "content": lesson.get("content"),
                "summary": lesson.get("summary"),
                "topics": lesson.get("topics", [])
            })
    finally:
        conn.close()


@mcp.tool()
def get_agent_decisions(user_id: str, agent_type: str = None, limit: int = 10) -> str:
    """
    Get audit log of platform agent decisions for this learner.
    Call ONLY when learner asks WHY a decision was made.
    agent_type: pathway_agent, semantic_rebuild_agent, code_review_agent, etc.
    """
    log.info(f"Getting agent decisions for user={user_id}, agent_type={agent_type or 'all'}, limit={limit}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if agent_type:
                cur.execute("""
                    SELECT * FROM agent_decisions
                    WHERE user_id = %s AND agent_type = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (user_id, agent_type, limit))
            else:
                cur.execute("""
                    SELECT * FROM agent_decisions
                    WHERE user_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (user_id, limit))
            
            decisions = cur.fetchall()
            
            log.info(f"Successfully retrieved {len(decisions)} agent decisions for user={user_id}")
            return json.dumps({
                "user_id": user_id,
                "agent_type": agent_type,
                "decisions": [dict(d) for d in decisions]
            }, default=str)
    finally:
        conn.close()


@mcp.tool()
def persist_conversation_message(
    conversation_id: str,
    session_id: str,
    user_id: str,
    role: str,
    message: str
) -> str:
    """
    Save a single message to conversation log immediately.
    Call TWICE per turn: once for learner's message (role='user'),
    once for your response (role='mentor').
    """
    log.info(f"Persisting conversation message: conversation_id={conversation_id}, role={role}")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            message_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO raw_mentor_chat_turns 
                (id, user_id, session_id, conversation_id, role, message, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (message_id, user_id, session_id, conversation_id, role, message))
            conn.commit()
            
            log.info(f"Successfully persisted message: message_id={message_id}, role={role}")
            return json.dumps({
                "status": "persisted",
                "message_id": message_id,
                "role": role
            })
    except Exception as e:
        conn.rollback()
        log.error(f"persist_conversation_message error: {e}")
        return json.dumps({"status": "error", "error": str(e)})
    finally:
        conn.close()


@mcp.tool()
def get_raw_session_transcript(session_id: str) -> str:
    """
    Get complete User:/Mentor: transcript for a finished session.
    FOR EPISODIC EXTRACTION AGENT USE ONLY - not for mentor chatbot.
    """
    log.info(f"Getting raw session transcript for session_id={session_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT role, message, created_at
                FROM raw_mentor_chat_turns
                WHERE session_id = %s
                ORDER BY created_at
            """, (session_id,))
            turns = cur.fetchall()
            
            if not turns:
                log.debug(f"No transcript found for session_id={session_id}")
                return json.dumps({
                    "session_id": session_id,
                    "transcript": "",
                    "line_count": 0
                })
            
            lines = []
            for turn in turns:
                role_label = "User" if turn["role"] == "user" else "Mentor"
                lines.append(f"{role_label}: {turn['message']}")
            
            transcript = "\n".join(lines)
            
            log.info(f"Successfully retrieved session transcript: session_id={session_id}, lines={len(lines)}")
            return json.dumps({
                "session_id": session_id,
                "transcript": transcript,
                "line_count": len(lines)
            })
    finally:
        conn.close()


@mcp.tool()
def get_current_course(user_id: str) -> str:
    """
    Get the course the learner is currently studying.
    Returns course_id, title, current_lesson_id.
    """
    log.info(f"Getting current course for user={user_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT sp.course_id, c.title, sp.current_lesson_id
                FROM student_pathways sp
                JOIN courses c ON sp.course_id = c.course_id
                WHERE sp.user_id = %s AND sp.status = 'in_progress'
                ORDER BY sp.started_at DESC
                LIMIT 1
            """, (user_id,))
            course = cur.fetchone()
            
            if not course:
                log.debug(f"No active course found for user={user_id}")
                return json.dumps({
                    "success": False,
                    "error": "No active course found"
                })
            
            log.info(f"Successfully retrieved current course for user={user_id}: {course['course_id']}")
            return json.dumps({
                "success": True,
                "course_id": course["course_id"],
                "title": course["title"],
                "current_lesson_id": course["current_lesson_id"]
            })
    finally:
        conn.close()


@mcp.tool()
def get_recent_exercises(user_id: str, course_id: str = None, limit: int = 5) -> str:
    """
    Get learner's recent exercise attempts with scores.
    Use to understand what they've been practicing recently.
    """
    log.info(f"Getting recent exercises for user={user_id}, course={course_id or 'all'}, limit={limit}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if course_id:
                cur.execute("""
                    SELECT * FROM exercise_attempts
                    WHERE user_id = %s AND course_id = %s
                    ORDER BY attempted_at DESC
                    LIMIT %s
                """, (user_id, course_id, limit))
            else:
                cur.execute("""
                    SELECT * FROM exercise_attempts
                    WHERE user_id = %s
                    ORDER BY attempted_at DESC
                    LIMIT %s
                """, (user_id, limit))
            
            exercises = cur.fetchall()
            
            log.info(f"Successfully retrieved {len(exercises)} recent exercises for user={user_id}")
            return json.dumps({
                "user_id": user_id,
                "course_id": course_id,
                "exercises": [dict(e) for e in exercises]
            }, default=str)
    finally:
        conn.close()


# =============================================================================
# VIVA AGENT TOOLS — Runtime viva examination support
# =============================================================================

@mcp.tool()
def get_capstone_details(user_id: str) -> str:
    """Get the mega capstone requirements, description, and related context.
    Use this to understand what the user was supposed to build."""
    log.info(f"Getting capstone details for user={user_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # First try to find via pathway
            cur.execute("""
                SELECT c.* FROM capstones c
                JOIN student_pathways sp ON c.capstone_id = sp.capstone_id
                WHERE sp.user_id = %s AND sp.status = 'completed'
                AND (c.capstone_id LIKE '%%mega%%' OR c.title ILIKE '%%mega%%')
                ORDER BY c.created_at DESC
                LIMIT 1
            """, (user_id,))
            capstone = cur.fetchone()
            
            if not capstone:
                # Fallback: find any mega capstone
                cur.execute("""
                    SELECT * FROM capstones
                    WHERE capstone_id LIKE '%%mega%%' OR title ILIKE '%%mega%%'
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                capstone = cur.fetchone()
            
            if not capstone:
                log.debug(f"No mega capstone found for user={user_id}")
                return json.dumps({
                    "success": False,
                    "error": "No mega capstone found for user",
                    "data": None
                })
            
            log.info(f"Successfully retrieved capstone details: {capstone['capstone_id']}")
            return json.dumps({
                "success": True,
                "error": None,
                "data": {
                    "capstone_id": capstone["capstone_id"],
                    "title": capstone["title"],
                    "description": capstone["description"],
                    "passing_score": capstone["passing_score"],
                }
            }, default=str)
    finally:
        conn.close()


@mcp.tool()
def get_capstone_review(user_id: str, capstone_id: str = None) -> str:
    """Get the code review results for the user's mega capstone submission.
    Includes code quality, design patterns, strengths, and areas for improvement."""
    log.info(f"Getting capstone review for user={user_id}, capstone={capstone_id or 'auto'}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get capstone ID if not provided
            if not capstone_id:
                cur.execute("""
                    SELECT capstone_id FROM capstones
                    WHERE capstone_id LIKE '%%mega%%' OR title ILIKE '%%mega%%'
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if not row:
                    return json.dumps({
                        "success": False,
                        "error": "No mega capstone found",
                        "data": None
                    })
                capstone_id = row["capstone_id"]
            
            # Get code review episode
            cur.execute("""
                SELECT * FROM episodic_episodes
                WHERE user_id = %s AND type = 'CAPSTONE_CODE_REVIEW'
                AND data->>'capstone_id' = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (user_id, capstone_id))
            review = cur.fetchone()
            
            if not review:
                return json.dumps({
                    "success": False,
                    "error": f"No code review found for capstone {capstone_id}",
                    "data": None
                })
            
            data = review["data"]
            if isinstance(data, str):
                data = json.loads(data)
            
            log.info(f"Successfully retrieved capstone review for user={user_id}: {capstone_id}")
            return json.dumps({
                "success": True,
                "error": None,
                "data": {
                    "episode_id": review["episode_id"],
                    "capstone_id": capstone_id,
                    "timestamp": review["timestamp"].isoformat() if hasattr(review["timestamp"], "isoformat") else str(review["timestamp"]),
                    "review": data,
                }
            }, default=str)
    finally:
        conn.close()


@mcp.tool()
def get_capstone_test(user_id: str, capstone_id: str = None) -> str:
    """Get the test results for the user's mega capstone submission.
    Includes pass/fail status, individual test cases, and any failures."""
    log.info(f"Getting capstone test results for user={user_id}, capstone={capstone_id or 'auto'}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Get capstone ID if not provided
            if not capstone_id:
                cur.execute("""
                    SELECT capstone_id FROM capstones
                    WHERE capstone_id LIKE '%%mega%%' OR title ILIKE '%%mega%%'
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if not row:
                    return json.dumps({
                        "success": False,
                        "error": "No mega capstone found",
                        "data": None
                    })
                capstone_id = row["capstone_id"]
            
            # Get test run episode
            cur.execute("""
                SELECT * FROM episodic_episodes
                WHERE user_id = %s AND type = 'CAPSTONE_TEST_RUN'
                AND data->>'capstone_id' = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (user_id, capstone_id))
            test_results = cur.fetchone()
            
            if not test_results:
                return json.dumps({
                    "success": False,
                    "error": f"No test results found for capstone {capstone_id}",
                    "data": None
                })
            
            data = test_results["data"]
            if isinstance(data, str):
                data = json.loads(data)
            
            log.info(f"Successfully retrieved capstone test results for user={user_id}: {capstone_id}")
            return json.dumps({
                "success": True,
                "error": None,
                "data": {
                    "episode_id": test_results["episode_id"],
                    "capstone_id": capstone_id,
                    "timestamp": test_results["timestamp"].isoformat() if hasattr(test_results["timestamp"], "isoformat") else str(test_results["timestamp"]),
                    "test_results": data,
                }
            }, default=str)
    finally:
        conn.close()


@mcp.tool()
def start_viva(user_id: str, capstone_id: str, pathway_id: str = None) -> str:
    """Start a new viva session for a user.
    Must be called before recording questions and responses.
    Returns session_id and attempt_id needed for recording conversation turns."""
    log.info(f"Starting viva session for user={user_id}, capstone={capstone_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Validate user
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            if not cur.fetchone():
                return json.dumps({
                    "success": False,
                    "error": f"User {user_id} not found",
                    "data": None
                })
            
            # Validate capstone
            cur.execute("SELECT * FROM capstones WHERE capstone_id = %s", (capstone_id,))
            if not cur.fetchone():
                return json.dumps({
                    "success": False,
                    "error": f"Capstone {capstone_id} not found",
                    "data": None
                })
            
            # Create session
            session_id = f"viva-{uuid.uuid4().hex[:12]}"
            attempt_id = f"attempt-{uuid.uuid4().hex[:12]}"
            
            # Store viva-specific data in device_info JSONB
            session_metadata = {
                "capstone_id": capstone_id,
                "pathway_id": pathway_id,
                "attempt_id": attempt_id,
                "status": "in_progress"
            }
            
            cur.execute("""
                INSERT INTO sessions (session_id, user_id, session_type, started_at, device_info)
                VALUES (%s, %s, 'capstone_viva', NOW(), %s)
                RETURNING *
            """, (session_id, user_id, json.dumps(session_metadata)))
            session = cur.fetchone()
            conn.commit()
            
            log.info(f"Successfully started viva session: session_id={session['session_id']}, attempt_id={attempt_id}")
            return json.dumps({
                "success": True,
                "error": None,
                "data": {
                    "session_id": session["session_id"],
                    "attempt_id": attempt_id,
                    "user_id": user_id,
                    "capstone_id": capstone_id,
                    "pathway_id": pathway_id,
                    "status": "in_progress",
                    "started_at": session["started_at"].isoformat() if session["started_at"] else None,
                }
            }, default=str)
    except Exception as e:
        conn.rollback()
        log.error(f"start_viva error: {e}")
        return json.dumps({"success": False, "error": str(e), "data": None})
    finally:
        conn.close()


@mcp.tool()
def get_viva_session(user_id: str, session_id: str) -> str:
    """Get the current state of a viva session including all conversation turns so far."""
    log.info(f"Getting viva session: session_id={session_id}, user={user_id}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM sessions
                WHERE session_id = %s AND session_type = 'capstone_viva'
            """, (session_id,))
            session = cur.fetchone()
            
            if not session:
                log.debug(f"Viva session {session_id} not found")
                return json.dumps({
                    "success": False,
                    "error": f"Viva session {session_id} not found",
                    "data": None
                })
            
            # Extract metadata from device_info
            metadata = session.get("device_info") or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            # Get all conversation turns
            cur.execute("""
                SELECT * FROM raw_viva_turns
                WHERE session_id = %s
                ORDER BY turn_number
            """, (session_id,))
            turns = cur.fetchall()
            
            log.info(f"Successfully retrieved viva session: session_id={session_id}, turns={len(turns)}")
            return json.dumps({
                "success": True,
                "error": None,
                "data": {
                    "session_id": session["session_id"],
                    "user_id": session["user_id"],
                    "session_type": session["session_type"],
                    "capstone_id": metadata.get("capstone_id"),
                    "pathway_id": metadata.get("pathway_id"),
                    "attempt_id": metadata.get("attempt_id"),
                    "status": metadata.get("status", "in_progress"),
                    "started_at": session["started_at"].isoformat() if session["started_at"] else None,
                    "ended_at": session["ended_at"].isoformat() if session["ended_at"] else None,
                    "turns": [dict(t) for t in turns],
                    "turn_count": len(turns),
                }
            }, default=str)
    finally:
        conn.close()


@mcp.tool()
def record_viva_turn(
    user_id: str,
    session_id: str,
    capstone_id: str,
    attempt_id: str,
    turn_number: int,
    role: str,
    message: str,
    pathway_id: str = None,
    episode_id: str = None,
    question_type: str = None,
    context_source: str = None,
    understanding_signals: dict = None
) -> str:
    """Record a conversation turn during the viva examination.
    Call this for EVERY message exchanged - both viva_agent questions and user responses.
    
    role: 'viva_agent' for your questions/statements, 'user' for the student's responses
    turn_number: sequential turn number starting from 1
    message: the full text of the message
    question_type: (optional) For viva_agent core questions: 'concept', 'code_specific', 'edge_case', 'debugging'
    context_source: (optional) For viva_agent questions: 'code_review', 'test_failure', 'design_pattern', 'requirement'
    understanding_signals: (optional) For user responses: dict with evaluation signals like {"clarity": "good", "depth": "moderate"}
    """
    log.info(f"Recording viva turn: session={session_id}, turn={turn_number}, role={role}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Build metadata JSON
            metadata = {}
            if question_type:
                metadata['question_type'] = question_type
            if context_source:
                metadata['context_source'] = context_source
            if understanding_signals:
                metadata['understanding_signals'] = understanding_signals
            
            # Note: Schema doesn't have metadata column, so store in message if needed
            # For now, just insert basic fields
            
            cur.execute("""
                INSERT INTO raw_viva_turns
                (user_id, session_id, capstone_id, pathway_id, attempt_id, turn_number, role, message, episode_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
            """, (user_id, session_id, capstone_id, pathway_id, attempt_id, turn_number, role, message, episode_id))
            turn = cur.fetchone()
            conn.commit()
            
            log.info(f"Successfully recorded viva turn: turn_id={turn['id']}, session={session_id}, turn={turn_number}")
            return json.dumps({
                "success": True,
                "error": None,
                "data": {
                    "turn_id": turn["id"],
                    "user_id": user_id,
                    "session_id": session_id,
                    "capstone_id": capstone_id,
                    "attempt_id": attempt_id,
                    "turn_number": turn_number,
                    "role": role,
                    "message": message,
                    "created_at": turn["created_at"].isoformat() if turn["created_at"] else None,
                }
            }, default=str)
    except Exception as e:
        conn.rollback()
        log.error(f"record_viva_turn error: {e}")
        return json.dumps({"success": False, "error": str(e), "data": None})
    finally:
        conn.close()


@mcp.tool()
def complete_viva(
    user_id: str,
    session_id: str,
    result: str,
    summary: str
) -> str:
    """Complete a viva session with the final pass/fail result and summary.
    
    result: 'pass' or 'fail'
    summary: Summary of the viva and reasoning for the decision
    """
    log.info(f"Completing viva session: session_id={session_id}, user={user_id}, result={result}")
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM sessions
                WHERE session_id = %s AND session_type = 'capstone_viva'
            """, (session_id,))
            session_before = cur.fetchone()
            
            if not session_before:
                return json.dumps({
                    "success": False,
                    "error": f"Viva session {session_id} not found",
                    "data": None
                })
            
            # Extract metadata
            metadata = session_before.get("device_info") or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            # Update metadata with completion info
            metadata["status"] = "completed"
            metadata["result"] = result
            
            # Complete session
            cur.execute("""
                UPDATE sessions
                SET ended_at = NOW(), device_info = %s
                WHERE session_id = %s
                RETURNING *
            """, (json.dumps(metadata), session_id))
            session = cur.fetchone()
            conn.commit()
            
            # Calculate duration
            started = session_before["started_at"]
            ended = session["ended_at"]
            if started and ended:
                duration_minutes = int((ended - started).total_seconds() / 60)
            else:
                duration_minutes = 0
            
            # Count questions asked (turns with role='viva_agent')
            cur.execute("""
                SELECT COUNT(*) as question_count
                FROM raw_viva_turns
                WHERE session_id = %s
                  AND role = 'viva_agent'
            """, (session_id,))
            q_count = cur.fetchone()
            questions_asked = q_count["question_count"] if q_count else 0
            
            # Store viva episode
            cur.execute("""
                INSERT INTO episodic_episodes (user_id, type, schema_version, data)
                VALUES (%s, 'CAPSTONE_VIVA', 1, %s)
                RETURNING episode_id
            """, (user_id, json.dumps({
                "session_id": session_id,
                "capstone_id": metadata.get("capstone_id"),
                "pathway_id": metadata.get("pathway_id"),
                "attempt_id": metadata.get("attempt_id"),
                "result": result,
                "summary": summary,
                "questions_asked": questions_asked,
                "duration_minutes": duration_minutes,
            })))
            episode = cur.fetchone()
            conn.commit()
            
            log.info(f"Successfully completed viva: session_id={session_id}, episode_id={episode['episode_id']}, result={result}")
            return json.dumps({
                "success": True,
                "error": None,
                "data": {
                    "session_id": session["session_id"],
                    "episode_id": episode["episode_id"],
                    "status": "completed",
                    "result": result,
                    "summary": summary,
                    "started_at": started.isoformat() if started else None,
                    "ended_at": ended.isoformat() if ended else None,
                    "questions_asked": questions_asked,
                    "duration_minutes": duration_minutes,
                }
            }, default=str)
    except Exception as e:
        conn.rollback()
        log.error(f"complete_viva error: {e}")
        return json.dumps({"success": False, "error": str(e), "data": None})
    finally:
        conn.close()


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))
    log.info("Starting Neulearn MCP Server (SSE) on %s:%s", host, port)
    mcp.run(transport="sse", host=host, port=port)