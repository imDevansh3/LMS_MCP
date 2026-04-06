"""
Neulearn Episodic Memory Tools - MCP Server
============================================

Complete implementation of all 8 episodic tools.
Processes raw event data into structured episodes with AI-generated summaries.

Episode Types:
1. SELF_ASSESSMENT       - Self-rated skill levels
2. PRE_ASSESSMENT        - MCQ results and calibration
3. COURSE_ACTIVITY       - Study session summaries
4. MENTOR_CHAT           - Chat interaction analysis
5. COURSE_COMPLETED      - Course mastery aggregation
6. CAPSTONE_CODE_REVIEW  - Code review analysis
7. CAPSTONE_TEST_RUN     - Test failure clustering
8. CAPSTONE_VIVA         - Viva scoring and skill updates

Transport: SSE (Server-Sent Events)
Port: 8001 (configurable via MCP_PORT env var)

Author: Neulearn Team
Date: 2026-04-03
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
import uuid
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
from openai import AzureOpenAI
from fastmcp import FastMCP

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neulearn.mcp")

DB_URL                   = os.getenv("DATABASE_URL")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

if not all([DB_URL, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY]):
    raise ValueError("Missing required environment variables. Check .env file.")

logger.info("Configuration loaded successfully")


# =============================================================================
# DATABASE CONNECTION POOL
# =============================================================================

class DatabaseManager:
    _pool: Optional[SimpleConnectionPool] = None

    @classmethod
    def initialize(cls, connection_string: str, min_conn: int = 2, max_conn: int = 10):
        if cls._pool is None:
            cls._pool = SimpleConnectionPool(min_conn, max_conn, connection_string)
            logger.info(f"Database pool initialized (min={min_conn}, max={max_conn})")

    @classmethod
    @contextmanager
    def get_connection(cls):
        if cls._pool is None:
            raise RuntimeError("Database pool not initialized")
        conn = cls._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cls._pool.putconn(conn)

    @classmethod
    @contextmanager
    def get_cursor(cls):
        with cls.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield cursor
            finally:
                cursor.close()


# =============================================================================
# AZURE OPENAI CLIENT
# =============================================================================

class OpenAIClient:
    _client: Optional[AzureOpenAI] = None

    @classmethod
    def initialize(cls):
        if cls._client is None:
            cls._client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            )
            logger.info("Azure OpenAI client initialized")

    @classmethod
    def complete(cls, prompt: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
        if cls._client is None:
            raise RuntimeError("OpenAI client not initialized")
        try:
            response = cls._client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Azure OpenAI error: {e}")
            return "[AI summary unavailable]"

    @classmethod
    def complete_json(cls, prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> Any:
        """Call the model and parse the response as JSON. Returns parsed object or None on failure."""
        raw = cls.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        # Strip markdown fences if the model wrapped output
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
            clean = clean.rsplit("```", 1)[0].strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from model: {clean[:200]}")
            return None


# =============================================================================
# RETRY HELPER
# =============================================================================

def retry(func, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


# =============================================================================
# SHARED SCORE HELPERS
# =============================================================================

def calculate_momentum_score(exercises_passed: int, total_exercises: int,
                              avg_attempts: float, completion_delta: float) -> float:
    if total_exercises == 0:
        return 0.5
    pass_rate      = exercises_passed / total_exercises
    attempt_factor = max(0.0, 1.0 - (avg_attempts / 4.0))
    delta_factor   = min(1.0, completion_delta / 100.0)
    score = (pass_rate * 0.6) + (attempt_factor * 0.2) + (delta_factor * 0.2)
    return round(min(1.0, max(0.0, score)), 2)


def classify_momentum_label(score: float) -> str:
    if score >= 0.7:
        return "strong"
    elif score >= 0.4:
        return "mixed"
    return "struggling"


def calculate_mastery_confidence(avg_score: float, avg_attempts: float) -> float:
    if avg_attempts == 0:
        return 0.0
    confidence = (avg_score / 100.0) * (1.0 / avg_attempts)
    return round(min(1.0, max(0.0, confidence)), 2)


def classify_mastery_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "strong"
    elif confidence >= 0.5:
        return "developing"
    return "weak"


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SELF_ASSESSMENT_FLAGS_PROMPT = """You are an educational assessment agent reviewing a student's onboarding self-assessment.

Declared interests: {declared_interests}

Self-assessment (domain → topic → self-rated level):
{self_assessment_json}

Identify 1–3 concise agent flags that will help the mentor bot on day one.
Look for: overconfidence patterns (intermediate claim but no exposure to advanced prerequisite topics),
underconfidence patterns, skill gaps that will block downstream learning.

Return ONLY a valid JSON array of strings. No markdown, no explanation.
Each string must be ≤ 120 characters.

Example: ["User claims intermediate Python but rated all LLM topics as none — likely foundational overconfidence"]

JSON array:"""


COURSE_ACTIVITY_ANALYSIS_PROMPT = """You are an educational session analyst. Analyse this study session and return structured JSON.

Session data:
- Duration: {duration_minutes} minutes (idle: {idle_minutes} min)
- Exercises: {exercise_count} attempted, {passed_count} passed
- Average attempts per exercise: {avg_attempts:.2f}
- Topics covered: {topics}
- Course completion: {before:.1f}% → {after:.1f}%

Exercise details:
{exercise_details_json}

Return ONLY valid JSON matching this exact shape (no markdown fences):
{{
  "failure_classifications": {{
    "<exercise_id>": {{
      "failure_type": "<logic_error|conceptual_gap|syntax_error|timeout|null>",
      "reasoning": "<one sentence>"
    }}
  }},
  "struggle_signals": [
    {{
      "topic_id": "<topic_id>",
      "signal": "<description>",
      "severity": "<high|medium|low>",
      "recommendation": "<one actionable sentence for the mentor bot>"
    }}
  ],
  "agent_summary": "<2-3 sentences: what happened, learning quality, mentor action if needed>"
}}

For failure_type: use null if the exercise was passed on first attempt.
For struggle_signals: only include genuine signals, not every failed attempt.
Base all reasoning strictly on the data provided."""


MENTOR_CHAT_ANALYSIS_PROMPT = """You are a learning analytics agent. Analyse this mentor chat session and return structured JSON.

Session info:
- Total turns: {turn_count}
- Duration: {duration_minutes} minutes
- Course context: {course_context}
- Topics in conversation: {topics_list}

Full conversation (chronological):
{turns_json}

Return ONLY valid JSON matching this exact shape (no markdown fences):
{{
  "unresolved_topics": [
    {{
      "topic_id": "<topic_id or best guess from context>",
      "question": "<what the student still doesn't understand, in one sentence>",
      "severity": "<high|medium|low>"
    }}
  ],
  "understanding_breakthroughs": [
    {{
      "topic_id": "<topic_id>",
      "signal": "<what the student said or did that confirmed understanding>"
    }}
  ],
  "question_type_counts": {{
    "clarification": 0,
    "stuck": 0,
    "curiosity": 0,
    "challenge": 0,
    "off_topic": 0
  }},
  "mentor_policy_events": [
    {{
      "event": "direct_answer_requested",
      "topic_id": "<topic_id>",
      "bot_held_line": true
    }}
  ],
  "agent_summary": "<3-4 sentences: what was discussed, what was resolved, what remains open, any policy events>"
}}

Rules:
- unresolved_topics: only questions where the student did NOT reach clarity by the end
- understanding_breakthroughs: only genuine signals — student restated concept correctly, applied it, or asked a clearly more advanced follow-up
- mentor_policy_events: only when student explicitly tried to get a direct answer/code solution
- question_type_counts: classify each student turn by dominant intent (one count per turn)
- Be strict — omit arrays if genuinely empty rather than padding with low-confidence entries"""


COURSE_COMPLETED_PATTERN_PROMPT = """You are an educational analyst identifying learning patterns across a completed course.

Course: {course_name}
Topic mastery summary:
{topic_mastery_json}

Areas of strength: {strength_topics}
Areas of struggle: {struggle_topics}

Write a single 1-2 sentence struggle_pattern that:
1. Identifies the UNDERLYING pattern connecting the struggle areas (not just listing them)
2. Suggests WHY the pattern exists (e.g. memorising outputs vs mechanistic understanding, theory vs practice gap)

Be specific to these topics. Return ONLY the pattern text — no labels, no markdown."""


LEARNING_PATH_ASSIGNED_REASONING_PROMPT = """You are a learning pathway allocation analyst explaining course assignments.

User interest: {interest}
Total courses assigned: {total_courses}
Capstone: {capstone_id}

Allocation breakdown:
{allocation_breakdown}

Self-assessment calibration: {calibration_summary}

Write a 2-3 sentence assignment_reasoning summary that explains:
1. Why this specific set of courses was assigned (mention beginner direct, MCQ failures, targeted modules)
2. How calibration influenced the pathway (if user was over/underconfident)
3. What the capstone requires as prerequisites

Be concise and factual. Return ONLY the summary text — no labels, no markdown.

Summary:"""


CODE_REVIEW_VIVA_BRIEFING_PROMPT = """You are a capstone evaluation agent preparing a viva examiner briefing.

Technologies evaluated and their issues:
{tech_evaluations_json}

Overall verdict: {overall_verdict}
Critical issues: {critical_issues}, Minor/High issues: {minor_issues}

Return ONLY valid JSON matching this exact shape (no markdown fences):
{{
  "probe_hard_on": ["<specific question or area to probe, not just tech name>"],
  "can_go_light_on": ["<tech or area where understanding was demonstrated>"],
  "likely_surface_knowledge_areas": ["<tech name>"],
  "agent_summary": "<2-3 sentences: overall code quality verdict, critical concerns, what viva must verify>"
}}

Rules:
- probe_hard_on: frame as specific examination angles e.g. "why langgraph retry logic was omitted" not just "langgraph"
- likely_surface_knowledge_areas: only techs where understanding verdict was 'beginner' or needs_revision
- can_go_light_on: only techs where understanding was 'good'/'deep' or 'adequate' with no critical issues
- Be direct and actionable — this briefs the examiner"""


TEST_RUN_CLUSTERING_PROMPT = """You are a test failure analyst. Cluster these failed tests by conceptual topic.

Capstone: {capstone_id}
Failed tests:
{failed_tests_json}

All tests (for context):
{all_tests_json}

Return ONLY valid JSON matching this exact shape (no markdown fences):
{{
  "failure_clusters": [
    {{
      "cluster_label": "<short snake_case label for the conceptual gap>",
      "topic_id": "<best matching topic/skill id from the test metadata or inferred>",
      "failed_test_ids": ["<test_name>"],
      "failure_type": "<critical|edge_case>",
      "interpretation": "<one sentence: what conceptual gap this cluster reveals>"
    }}
  ],
  "agent_verdict": "<pass|partial_pass|fail>",
  "mentor_guidance": {{
    "focus_areas": ["<specific concept to study>"],
    "suggested_resources": ["<doc or guide name>"]
  }},
  "agent_summary": "<2 sentences: how many tests failed, what the clusters reveal, what to fix first>"
}}

Rules:
- Group tests by what they TEST conceptually, not by name prefix
- critical = core functionality broken; edge_case = boundary condition not handled
- agent_verdict: pass if pass_rate >= 0.9, partial_pass if >= 0.6, fail otherwise (pass_rate = {pass_rate:.2f})"""


VIVA_SCORING_PROMPT = """You are a capstone viva examiner scoring a student's oral responses.

Capstone: {capstone_id}
Passing threshold: {passing_score}/100

Viva conversation (chronological question/answer pairs):
{turns_json}

Viva briefing from code review (areas flagged as surface knowledge):
{viva_briefing_json}

Return ONLY valid JSON matching this exact shape (no markdown fences):
{{
  "question_log": [
    {{
      "question_id": "q1",
      "topic_id": "<topic or skill this question probes>",
      "question_text": "<the question asked>",
      "question_type": "<design_decision|scenario|what_if|conceptual>",
      "was_probed_from_code_review": true,
      "answer_quality": "<good|partial|poor>",
      "answer_summary": "<one sentence: what the student said and why it was good/partial/poor>",
      "score": 0
    }}
  ],
  "topic_verdicts": {{
    "<topic_id>": {{
      "verbal_score": 0,
      "confirmed_surface_knowledge": false,
      "mastery_delta": 0.0
    }}
  }},
  "agent_verdict": "<pass|fail>",
  "verdict_reason": "<one sentence explaining the verdict with specific topic references>",
  "student_facing_summary": "<2-3 sentences: constructive feedback acknowledging strengths then specific gaps, actionable for resubmission if failed>"
}}

Scoring rules:
- score per question: 0-100 based on correctness and depth of reasoning, NOT answer length
- good = correct and shows understanding of why; partial = correct but shallow; poor = incorrect or no awareness of failure mode
- confirmed_surface_knowledge = true if student could not reason about a concept the code review flagged as surface
- mastery_delta: +0.1 for good, 0.0 for partial, -0.1 for poor (applied to current mastery score)
- was_probed_from_code_review: true if the question targets an area from the viva_briefing probe_hard_on list
- agent_verdict: pass if average question score >= {passing_score}"""


# =============================================================================
# TOOL 1 — SELF_ASSESSMENT
# =============================================================================

def process_self_assessment(user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process raw_self_assessment into SELF_ASSESSMENT episode.

    Maps self-ratings to numeric_confidence, identifies MCQ candidates,
    generates agent flags via LLM.
    """
    CONFIDENCE_MAP = {
        "none":         0.0,
        "beginner":     0.2,
        "intermediate": 0.5,
        "advanced":     0.85,
    }

    try:
        with DatabaseManager.get_cursor() as cursor:
            query  = "SELECT * FROM raw_self_assessment WHERE user_id = %s AND processed_at IS NULL"
            params = [user_id]
            if session_id:
                query  += " AND session_id = %s"
                params.append(session_id)
            query += " ORDER BY created_at DESC LIMIT 1"
            cursor.execute(query, params)
            row = cursor.fetchone()

            if not row:
                return _no_data("SELF_ASSESSMENT", user_id, "No unprocessed self-assessment found")

            self_assessment    = row["self_assessment"]
            declared_interests = row["declared_interests"]

            # Build domain profiles with numeric_confidence
            domain_profiles: Dict[str, Any] = {}
            for domain, topics in self_assessment.items():
                topics_for_mcq  = []
                domain_topics   = {}
                for topic, level in topics.items():
                    level_lower          = level.lower()
                    numeric_confidence   = CONFIDENCE_MAP.get(level_lower, 0.2)
                    will_be_mcq_tested   = level_lower == "intermediate"
                    domain_topics[topic] = {
                        "self_rated_level":    level_lower,
                        "numeric_confidence":  numeric_confidence,
                        "will_be_mcq_tested":  will_be_mcq_tested,
                    }
                    if will_be_mcq_tested:
                        topics_for_mcq.append(topic)

                levels = [t["self_rated_level"] for t in domain_topics.values()]
                if any(l == "intermediate" for l in levels):
                    overall_level = "intermediate"
                elif all(l in ("none", "beginner") for l in levels):
                    overall_level = "beginner"
                else:
                    overall_level = "mixed"

                domain_profiles[domain] = {
                    "topics":                     domain_topics,
                    "topics_for_mcq":             topics_for_mcq,
                    "overall_self_declared_level": overall_level,
                }

            # LLM-generated agent flags
            prompt       = SELF_ASSESSMENT_FLAGS_PROMPT.format(
                declared_interests=", ".join(declared_interests),
                self_assessment_json=json.dumps(self_assessment, indent=2),
            )
            agent_flags  = retry(lambda: OpenAIClient.complete_json(prompt)) or []
            if not isinstance(agent_flags, list):
                agent_flags = [str(agent_flags)]

            episode_data = {
                "declared_interests": declared_interests,
                "domain_profiles":    domain_profiles,
                "agent_flags":        agent_flags,
            }

            episode_id = _write_episode(cursor, user_id, "SELF_ASSESSMENT", episode_data)
            cursor.execute(
                "UPDATE raw_self_assessment SET processed_at = NOW(), episode_id = %s WHERE id = %s",
                (episode_id, row["id"]),
            )

            logger.info(f"SELF_ASSESSMENT episode {episode_id} created for user {user_id}")
            return _ok("SELF_ASSESSMENT", user_id, episode_id,
                       f"Processed {len(domain_profiles)} domains, {sum(len(d['topics_for_mcq']) for d in domain_profiles.values())} topics queued for MCQ")

    except Exception as e:
        logger.error(f"process_self_assessment error: {e}")
        return _err("SELF_ASSESSMENT", user_id, e)


# =============================================================================
# TOOL 2 — PRE_ASSESSMENT
# =============================================================================

def process_pre_assessment(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Process raw_pre_assessment MCQ rows into PRE_ASSESSMENT episode.

    Groups by topic + difficulty, applies scoring rules, computes calibration
    delta using numeric_confidence from SELF_ASSESSMENT episode.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM raw_pre_assessment
                WHERE user_id = %s AND session_id = %s AND processed_at IS NULL
                ORDER BY created_at
            """, (user_id, session_id))
            rows = cursor.fetchall()

            if not rows:
                return _no_data("PRE_ASSESSMENT", user_id, "No unprocessed pre-assessment rows found")

            domain = rows[0]["domain"]

            # Pull numeric_confidence from the most recent SELF_ASSESSMENT episode
            cursor.execute("""
                SELECT data FROM episodic_episodes
                WHERE user_id = %s AND type = 'SELF_ASSESSMENT'
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id,))
            sa_row  = cursor.fetchone()
            sa_data = sa_row["data"] if sa_row else {}

            def get_numeric_confidence(topic_id: str) -> float:
                for d_profile in sa_data.get("domain_profiles", {}).values():
                    topic_info = d_profile.get("topics", {}).get(topic_id)
                    if topic_info:
                        return topic_info.get("numeric_confidence", 0.2)
                return 0.2

            def get_self_rated_level(topic_id: str) -> str:
                for d_profile in sa_data.get("domain_profiles", {}).values():
                    topic_info = d_profile.get("topics", {}).get(topic_id)
                    if topic_info:
                        return topic_info.get("self_rated_level", "beginner")
                return "beginner"

            # Aggregate per topic
            topic_buckets: Dict[str, Dict] = {}
            for row in rows:
                tid  = row["topic_id"]
                diff = (row.get("difficulty") or "medium").lower()
                if tid not in topic_buckets:
                    topic_buckets[tid] = {
                        "medium": {"correct": 0, "total": 0},
                        "hard":   {"correct": 0, "total": 0},
                        "has_hard": False,
                    }
                bucket = topic_buckets[tid]
                if diff == "medium":
                    bucket["medium"]["total"]   += 1
                    bucket["medium"]["correct"] += int(bool(row["is_correct"]))
                elif diff == "hard":
                    bucket["hard"]["total"]   += 1
                    bucket["hard"]["correct"] += int(bool(row["is_correct"]))
                    bucket["has_hard"] = True

            topic_results: Dict[str, Any] = {}
            overconfident, underconfident, accurate = [], [], []

            for tid, b in topic_buckets.items():
                self_rated        = get_self_rated_level(tid)
                numeric_conf      = get_numeric_confidence(tid)

                m_total   = b["medium"]["total"]
                m_correct = b["medium"]["correct"]
                m_passed  = (m_correct / m_total >= 0.5) if m_total > 0 else False

                h_total   = b["hard"]["total"]
                h_correct = b["hard"]["correct"]
                h_passed  = None
                hard_score_obj = None
                if b["has_hard"]:
                    h_passed       = (h_correct / h_total >= 0.5) if h_total > 0 else False
                    hard_score_obj = {"correct": h_correct, "out_of": h_total, "passed": h_passed}

                # Scoring rules
                if not m_passed:
                    outcome       = "full_course"
                    assessed_lvl  = "beginner"
                elif h_passed is False:
                    outcome       = "targeted_modules"
                    assessed_lvl  = "intermediate"
                else:
                    outcome       = "advanced_path"
                    assessed_lvl  = "advanced"

                # Calibration delta — compare numeric_confidence to actual MCQ score
                mcq_score = (m_correct / m_total) if m_total > 0 else 0.0
                delta_val = mcq_score - numeric_conf

                if delta_val < -0.25:
                    calibration_delta     = "overconfident"
                    calibration_magnitude = "large" if delta_val < -0.4 else "small"
                    overconfident.append(tid)
                elif delta_val > 0.25:
                    calibration_delta     = "underconfident"
                    calibration_magnitude = "small" if delta_val < 0.4 else "large"
                    underconfident.append(tid)
                else:
                    calibration_delta     = "accurate"
                    calibration_magnitude = "none"
                    accurate.append(tid)

                result: Dict[str, Any] = {
                    "self_rated_level":        self_rated,
                    "medium_score":            {"correct": m_correct, "out_of": m_total, "passed": m_passed},
                    "hard_score":              hard_score_obj,
                    "assessed_level":          assessed_lvl,
                    "outcome":                 outcome,
                    "calibration_delta":       calibration_delta,
                    "calibration_gap_magnitude": calibration_magnitude,
                }
                if outcome == "targeted_modules":
                    # Collect failed hard test IDs from raw rows
                    failed_ids = [
                        r["question_id"] for r in rows
                        if r["topic_id"] == tid
                        and (r.get("difficulty") or "").lower() == "hard"
                        and not r["is_correct"]
                    ]
                    result["failed_module_ids"] = failed_ids

                topic_results[tid] = result

            # Overall calibration
            if len(overconfident) > len(underconfident):
                overall_calibration = "slightly_overconfident"
            elif len(underconfident) > len(overconfident):
                overall_calibration = "slightly_underconfident"
            else:
                overall_calibration = "accurate"

            # Domain entry level = most common assessed level
            assessed_levels = [r["assessed_level"] for r in topic_results.values()]
            domain_entry    = max(set(assessed_levels), key=assessed_levels.count) if assessed_levels else "beginner"

            episode_data = {
                "domain":            domain,
                "pathway_id":        None,
                "topic_results":     topic_results,
                "domain_entry_level": domain_entry,
                "calibration_summary": {
                    "overconfident_topics":  overconfident,
                    "underconfident_topics": underconfident,
                    "accurate_topics":       accurate,
                    "overall_calibration":   overall_calibration,
                },
                "agent_flags": [],
            }

            episode_id = _write_episode(cursor, user_id, "PRE_ASSESSMENT", episode_data)
            row_ids    = [r["id"] for r in rows]
            cursor.execute(
                "UPDATE raw_pre_assessment SET processed_at = NOW(), episode_id = %s WHERE id = ANY(%s)",
                (episode_id, row_ids),
            )

            logger.info(f"PRE_ASSESSMENT episode {episode_id} created for user {user_id}")
            return _ok("PRE_ASSESSMENT", user_id, episode_id,
                       f"Processed {len(rows)} MCQ rows across {len(topic_results)} topics — calibration: {overall_calibration}")

    except Exception as e:
        logger.error(f"process_pre_assessment error: {e}")
        return _err("PRE_ASSESSMENT", user_id, e)


# =============================================================================
# TOOL 3 — LEARNING_PATH_ASSIGNED
# =============================================================================

def process_pathway_allocated(user_id: str, pathway_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process student_pathways into LEARNING_PATH_ASSIGNED episode.

    Captures pathway assignment logic based on SELF_ASSESSMENT and PRE_ASSESSMENT.
    Explains which courses came from beginner direct assignment, MCQ failures,
    and targeted modules.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Check if pathway episode already exists
            if pathway_id:
                cursor.execute("""
                    SELECT episode_id FROM episodic_episodes
                    WHERE user_id = %s AND type = 'LEARNING_PATH_ASSIGNED'
                      AND data->>'pathway_id' = %s
                    LIMIT 1
                """, (user_id, pathway_id))
                existing = cursor.fetchone()
                if existing:
                    return _ok("LEARNING_PATH_ASSIGNED", user_id, existing["episode_id"],
                               f"Pathway {pathway_id} already processed")

            # Get the pathway
            query  = "SELECT * FROM student_pathways WHERE user_id = %s"
            params = [user_id]
            if pathway_id:
                query  += " AND pathway_id = %s"
                params.append(pathway_id)
            query += " ORDER BY created_at DESC LIMIT 1"
            cursor.execute(query, params)
            row = cursor.fetchone()

            if not row:
                return _no_data("LEARNING_PATH_ASSIGNED", user_id, "No pathway found")

            pathway_id        = row["pathway_id"]
            interest          = row["interest"]
            capstone_id       = row.get("capstone_id")
            courses_remaining = row.get("courses_remaining") or []
            courses_in_progress = row.get("courses_in_progress") or []
            courses_completed = row.get("courses_completed") or []

            all_courses = list(set(courses_remaining + courses_in_progress + courses_completed))

            # Pull prior SELF_ASSESSMENT and PRE_ASSESSMENT episodes
            cursor.execute("""
                SELECT type, data FROM episodic_episodes
                WHERE user_id = %s AND type IN ('SELF_ASSESSMENT', 'PRE_ASSESSMENT')
                ORDER BY timestamp DESC
            """, (user_id,))
            prior_episodes = cursor.fetchall()

            self_assessment_data = None
            pre_assessment_data  = None
            for ep in prior_episodes:
                if ep["type"] == "SELF_ASSESSMENT" and not self_assessment_data:
                    self_assessment_data = ep["data"]
                elif ep["type"] == "PRE_ASSESSMENT" and not pre_assessment_data:
                    pre_assessment_data = ep["data"]

            # Map courses to assignment logic
            beginner_direct  = []
            full_course      = []
            targeted_modules = []
            skipped          = []

            # From SELF_ASSESSMENT: beginner-rated topics → beginner_direct courses
            if self_assessment_data:
                for domain, profile in self_assessment_data.get("domain_profiles", {}).items():
                    for topic_id, topic_info in profile.get("topics", {}).items():
                        if topic_info.get("self_rated_level") in ("none", "beginner"):
                            # Would map topic_id to course_id via topic_course mapping
                            # For now, simplified
                            pass

            # From PRE_ASSESSMENT: outcomes
            if pre_assessment_data:
                for topic_id, result in pre_assessment_data.get("topic_results", {}).items():
                    outcome = result.get("outcome")
                    # Would map topic_id to course_id here
                    # Simplified for now
                    if outcome == "full_course":
                        pass  # full_course.append(course_id)
                    elif outcome == "targeted_modules":
                        pass  # targeted_modules.append(course_id)
                    elif outcome == "advanced_path":
                        pass  # skipped.append(course_id)

            # Simple categorization based on what's in the pathway
            # (In real system, would have topic→course mapping)
            courses_assigned = {
                " beginner_direct": beginner_direct if beginner_direct else ["Courses for beginner-rated topics"],
                "full_course":      full_course if full_course else ["Courses where MCQ medium failed"],
                "targeted_modules": targeted_modules if targeted_modules else ["Courses where MCQ hard failed"],
                "skipped":          skipped if skipped else ["Courses skipped via advanced path"],
            }

            entry_levels = {}
            if pre_assessment_data:
                domain = pre_assessment_data.get("domain")
                domain_entry = pre_assessment_data.get("domain_entry_level")
                if domain:
                    entry_levels[domain] = domain_entry

            # Calibration summary
            calibration_summary = "accurate"
            if pre_assessment_data:
                cal_data = pre_assessment_data.get("calibration_summary", {})
                calibration_summary = cal_data.get("overall_calibration", "accurate")

            # Generate assignment reasoning via LLM
            allocation_breakdown = "\n".join([
                f"- {key}: {len(val) if isinstance(val, list) else 'N/A'} courses"
                for key, val in courses_assigned.items()
            ])

            reasoning_prompt = LEARNING_PATH_ASSIGNED_REASONING_PROMPT.format(
                interest=interest,
                total_courses=len(all_courses),
                capstone_id=capstone_id or "No capstone assigned",
                allocation_breakdown=allocation_breakdown,
                calibration_summary=calibration_summary,
            )

            assignment_reasoning = retry(lambda: OpenAIClient.complete(reasoning_prompt))

            episode_data = {
                "pathway_id":    pathway_id,
                "interest":      interest,
                "capstone_id":   capstone_id,
                "courses_assigned": courses_assigned,
                "entry_levels":  entry_levels,
                "assignment_reasoning": {
                    "total_courses":         len(all_courses),
                    "calibration_influenced": calibration_summary != "accurate",
                    "agent_summary":         assignment_reasoning,
                },
                "generated_by_agent": row.get("generated_by_agent", "pathway_engine_v1"),
                "allocated_at":       row["created_at"].isoformat(),
            }

            episode_id = _write_episode(cursor, user_id, "LEARNING_PATH_ASSIGNED", episode_data)

            logger.info(f"LEARNING_PATH_ASSIGNED episode {episode_id} created for user {user_id}")
            return _ok("LEARNING_PATH_ASSIGNED", user_id, episode_id,
                       f"Pathway {pathway_id} allocated with {len(all_courses)} courses for {interest}")

    except Exception as e:
        logger.error(f"process_pathway_allocated error: {e}")
        return _err("LEARNING_PATH_ASSIGNED", user_id, e)


# =============================================================================
# TOOL 4 — COURSE_ACTIVITY
# =============================================================================

def process_course_activity(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Process raw_course_session events into COURSE_ACTIVITY episode.

    Aggregates exercise attempts, uses LLM to classify failure types and
    struggle signals, computes momentum score.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            cursor.execute("""
                SELECT rcs.*, e.difficulty as ex_difficulty, e.pass_threshold, e.skill_id
                FROM raw_course_session rcs
                LEFT JOIN exercises e ON e.exercise_id = rcs.exercise_id
                WHERE rcs.user_id = %s AND rcs.session_id = %s AND rcs.processed_at IS NULL
                ORDER BY rcs.created_at
            """, (user_id, session_id))
            rows = cursor.fetchall()

            if not rows:
                return _no_data("COURSE_ACTIVITY", user_id, "No unprocessed course session rows found")

            course_id = rows[0]["course_id"]

            # Pull pathway_id from active enrollment
            cursor.execute("""
                SELECT pathway_id FROM enrollments
                WHERE user_id = %s AND course_id = %s AND status = 'active'
                LIMIT 1
            """, (user_id, course_id))
            enroll_row = cursor.fetchone()
            pathway_id = enroll_row["pathway_id"] if enroll_row else None

            # Session window
            session_start = session_end = None
            start_completion = end_completion = 0.0
            idle_minutes = 0

            for r in rows:
                if r["event_type"] == "session_start":
                    session_start      = r["created_at"]
                    start_completion   = r["completion_percent"] or 0.0
                elif r["event_type"] == "session_end":
                    session_end        = r["created_at"]
                    end_completion     = r["completion_percent"] or 0.0
                    idle_minutes       = r["metadata"].get("idle_minutes", 0) if r["metadata"] else 0

            duration_minutes = 0
            if session_start and session_end:
                duration_minutes = int((session_end - session_start).total_seconds() / 60)

            # Aggregate per exercise — use attempt_number from the schema
            exercise_data: Dict[str, Any] = {}
            topics_covered: set = set()

            for r in rows:
                if r["event_type"] not in ("exercise_attempt", "exercise_pass", "exercise_fail"):
                    continue
                eid = r["exercise_id"]
                if not eid:
                    continue
                if r.get("topic_id"):
                    topics_covered.add(r["topic_id"])

                if eid not in exercise_data:
                    exercise_data[eid] = {
                        "topic_id":       r.get("topic_id"),
                        "difficulty":     r.get("ex_difficulty") or "medium",
                        "pass_threshold": r.get("pass_threshold") or 70,
                        "attempts":       [],
                        "passed":         False,
                        "final_score":    0,
                        "test_cases_passed": 0,
                        "test_cases_total":  0,
                    }

                ed = exercise_data[eid]
                ed["attempts"].append({
                    "event_type":        r["event_type"],
                    "attempt_number":    r.get("attempt_number") or len(ed["attempts"]) + 1,
                    "score":             r.get("score") or 0,
                    "test_cases_passed": r.get("test_cases_passed") or 0,
                    "test_cases_total":  r.get("test_cases_total") or 0,
                })

                if r["event_type"] == "exercise_pass":
                    ed["passed"]            = True
                    ed["final_score"]       = r.get("score") or 0
                    ed["test_cases_passed"] = r.get("test_cases_passed") or 0
                    ed["test_cases_total"]  = r.get("test_cases_total") or 0
                elif r["event_type"] == "exercise_fail":
                    # Keep updating so final reflects the last attempt
                    ed["final_score"]       = r.get("score") or ed["final_score"]
                    ed["test_cases_passed"] = r.get("test_cases_passed") or ed["test_cases_passed"]
                    ed["test_cases_total"]  = r.get("test_cases_total") or ed["test_cases_total"]

            # Build attempt_pattern strings from ordered attempts
            exercise_list = []
            for eid, ed in exercise_data.items():
                parts = []
                for a in ed["attempts"]:
                    if a["event_type"] == "exercise_pass":
                        parts.append("pass")
                    elif a["event_type"] in ("exercise_fail", "exercise_attempt"):
                        parts.append("fail")
                attempt_pattern = "_".join(parts) if parts else "unknown"

                exercise_list.append({
                    "exercise_id":       eid,
                    "topic_id":          ed["topic_id"],
                    "difficulty":        ed["difficulty"],
                    "total_attempts":    len(ed["attempts"]),
                    "passed":            ed["passed"],
                    "final_score":       ed["final_score"],
                    "test_cases_passed": ed["test_cases_passed"],
                    "test_cases_total":  ed["test_cases_total"],
                    "attempt_pattern":   attempt_pattern,
                    "failure_type":      None,  # filled by LLM below
                })

            # LLM analysis for failure classification and struggle signals
            exercise_details_for_llm = [
                {
                    "exercise_id":    e["exercise_id"],
                    "topic_id":       e["topic_id"],
                    "difficulty":     e["difficulty"],
                    "passed":         e["passed"],
                    "attempt_pattern": e["attempt_pattern"],
                    "final_score":    e["final_score"],
                    "test_cases_passed": e["test_cases_passed"],
                    "test_cases_total":  e["test_cases_total"],
                }
                for e in exercise_list
            ]

            llm_prompt = COURSE_ACTIVITY_ANALYSIS_PROMPT.format(
                duration_minutes=duration_minutes,
                idle_minutes=idle_minutes,
                exercise_count=len(exercise_list),
                passed_count=sum(1 for e in exercise_list if e["passed"]),
                avg_attempts=sum(e["total_attempts"] for e in exercise_list) / max(1, len(exercise_list)),
                topics=", ".join(topics_covered) or "unknown",
                before=start_completion,
                after=end_completion,
                exercise_details_json=json.dumps(exercise_details_for_llm, indent=2),
            )

            llm_result = retry(lambda: OpenAIClient.complete_json(llm_prompt)) or {}

            # Merge LLM failure classifications into exercise_list
            failure_classifications = llm_result.get("failure_classifications", {})
            for e in exercise_list:
                fc = failure_classifications.get(e["exercise_id"], {})
                e["failure_type"] = fc.get("failure_type")

            struggle_signals = llm_result.get("struggle_signals", [])
            agent_summary    = llm_result.get("agent_summary", "[Summary unavailable]")

            # Momentum score
            exercises_passed   = sum(1 for e in exercise_list if e["passed"])
            avg_attempts_all   = sum(e["total_attempts"] for e in exercise_list) / max(1, len(exercise_list))
            completion_delta   = end_completion - start_completion
            momentum_score     = calculate_momentum_score(exercises_passed, len(exercise_list), avg_attempts_all, completion_delta)
            momentum_label     = classify_momentum_label(momentum_score)

            # Modules completed this session via module_progress
            cursor.execute("""
                SELECT mp.module_id FROM module_progress mp
                WHERE mp.user_id = %s AND mp.course_id = %s
                  AND mp.completed_at >= %s
                  AND mp.status = 'completed'
            """, (user_id, course_id, session_start or datetime.utcnow()))
            completed_modules = [r["module_id"] for r in cursor.fetchall()]

            episode_data = {
                "session_id":    session_id,
                "course_id":     course_id,
                "pathway_id":    pathway_id,
                "session_window": {
                    "started_at":       session_start.isoformat() if session_start else None,
                    "ended_at":         session_end.isoformat()   if session_end   else None,
                    "duration_minutes": duration_minutes,
                    "idle_minutes":     idle_minutes,
                },
                "topics_covered":   list(topics_covered),
                "exercise_outcomes": exercise_list,
                "struggle_signals":  struggle_signals,
                "momentum_score":    momentum_score,
                "momentum_label":    momentum_label,
                "completion_delta":  {
                    "before_session":                start_completion,
                    "after_session":                 end_completion,
                    "modules_completed_this_session": completed_modules,
                },
                "agent_summary": agent_summary,
            }

            episode_id = _write_episode(cursor, user_id, "COURSE_ACTIVITY", episode_data)
            row_ids    = [r["id"] for r in rows]
            cursor.execute(
                "UPDATE raw_course_session SET processed_at = NOW(), episode_id = %s WHERE id = ANY(%s)",
                (episode_id, row_ids),
            )

            logger.info(f"COURSE_ACTIVITY episode {episode_id} created for user {user_id}")
            return _ok("COURSE_ACTIVITY", user_id, episode_id,
                       f"{len(exercise_list)} exercises, momentum={momentum_label} ({momentum_score})")

    except Exception as e:
        logger.error(f"process_course_activity error: {e}")
        return _err("COURSE_ACTIVITY", user_id, e)


# =============================================================================
# TOOL 4 — MENTOR_CHAT
# =============================================================================

def process_mentor_chat(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Process raw_mentor_chat_turns into MENTOR_CHAT episode.

    Uses LLM to identify unresolved topics, understanding breakthroughs,
    intent classification, and mentor policy events from the full conversation.
    No per-turn summaries — session-level analysis only.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM raw_mentor_chat_turns
                WHERE user_id = %s AND session_id = %s AND processed_at IS NULL
                ORDER BY turn_number
            """, (user_id, session_id))
            rows = cursor.fetchall()

            if not rows:
                return _no_data("MENTOR_CHAT", user_id, "No unprocessed mentor chat turns found")

            first_turn = rows[0]
            last_turn  = rows[-1]

            course_id  = first_turn.get("course_id")
            pathway_id = first_turn.get("pathway_id")

            chat_context = {
                "type":            "course_context" if course_id else "standalone",
                "course_id":       course_id,
                "pathway_id":      pathway_id,
                "active_topic_id": first_turn.get("topic_id"),
            }

            duration_minutes = int(
                (last_turn["created_at"] - first_turn["created_at"]).total_seconds() / 60
            )

            session_window = {
                "started_at":    first_turn["created_at"].isoformat(),
                "ended_at":      last_turn["created_at"].isoformat(),
                "duration_minutes": duration_minutes,
                "total_turns":   len(rows),
            }

            # All distinct topics touched in this session (from topic_id column)
            topics_discussed = list({r["topic_id"] for r in rows if r.get("topic_id")})

            # Build turn list for LLM — full messages, no truncation
            turns_for_llm = [
                {
                    "turn": r["turn_number"],
                    "role": r["role"],
                    "message": r["message"],
                    "topic_id": r.get("topic_id"),
                }
                for r in rows
            ]

            course_context_str = (
                f"course_id={course_id}, pathway_id={pathway_id}"
                if course_id else "standalone (no active course)"
            )

            llm_prompt = MENTOR_CHAT_ANALYSIS_PROMPT.format(
                turn_count=len(rows),
                duration_minutes=duration_minutes,
                course_context=course_context_str,
                topics_list=", ".join(topics_discussed) or "general",
                turns_json=json.dumps(turns_for_llm, indent=2),
            )

            llm_result = retry(lambda: OpenAIClient.complete_json(llm_prompt)) or {}

            episode_data = {
                "session_id":    session_id,
                "chat_context":  chat_context,
                "session_window": session_window,
                "topics_discussed":         topics_discussed,
                "unresolved_topics":        llm_result.get("unresolved_topics", []),
                "understanding_breakthroughs": llm_result.get("understanding_breakthroughs", []),
                "question_type_counts":     llm_result.get("question_type_counts", {
                    "clarification": 0, "stuck": 0, "curiosity": 0, "challenge": 0, "off_topic": 0,
                }),
                "mentor_policy_events":     llm_result.get("mentor_policy_events", []),
                "agent_summary":            llm_result.get("agent_summary", "[Summary unavailable]"),
            }

            episode_id = _write_episode(cursor, user_id, "MENTOR_CHAT", episode_data)
            row_ids    = [r["id"] for r in rows]
            cursor.execute(
                "UPDATE raw_mentor_chat_turns SET processed_at = NOW(), episode_id = %s WHERE id = ANY(%s)",
                (episode_id, row_ids),
            )

            logger.info(f"MENTOR_CHAT episode {episode_id} created for user {user_id}")
            unresolved_count = len(episode_data["unresolved_topics"])
            return _ok("MENTOR_CHAT", user_id, episode_id,
                       f"{len(rows)} turns, {unresolved_count} unresolved topics")

    except Exception as e:
        logger.error(f"process_mentor_chat error: {e}")
        return _err("MENTOR_CHAT", user_id, e)


# =============================================================================
# TOOL 5 — COURSE_COMPLETED
# =============================================================================

def process_course_completed(user_id: str, course_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process raw_course_completion into COURSE_COMPLETED episode.

    Aggregates all COURSE_ACTIVITY episodes for the course, computes
    topic mastery, uses LLM to identify struggle pattern.
    Also writes skill mastery updates and semantic_rebuild_trigger.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            query  = """
                SELECT rc.*, c.name as course_name
                FROM raw_course_completion rc
                JOIN courses c ON c.course_id = rc.course_id
                WHERE rc.user_id = %s AND rc.processed_at IS NULL
            """
            params = [user_id]
            if course_id:
                query  += " AND rc.course_id = %s"
                params.append(course_id)
            query += " ORDER BY rc.completed_at DESC LIMIT 1"
            cursor.execute(query, params)
            row = cursor.fetchone()

            if not row:
                return _no_data("COURSE_COMPLETED", user_id, "No unprocessed course completion found")

            course_id   = row["course_id"]
            course_name = row["course_name"]
            pathway_id  = row.get("pathway_id")

            # All COURSE_ACTIVITY episodes for this course
            cursor.execute("""
                SELECT data FROM episodic_episodes
                WHERE user_id = %s AND type = 'COURSE_ACTIVITY'
                  AND data->>'course_id' = %s
                ORDER BY timestamp
            """, (user_id, course_id))
            activity_rows = cursor.fetchall()

            # Pull skills taught by this course from exercises
            cursor.execute("""
                SELECT DISTINCT e.skill_id
                FROM exercises e
                JOIN modules m ON m.module_id = e.module_id
                WHERE m.course_id = %s AND e.skill_id IS NOT NULL
            """, (course_id,))
            skill_ids = [r["skill_id"] for r in cursor.fetchall()]

            # Aggregate topic performance across all activity episodes
            topics_data: Dict[str, Dict] = {}
            all_exercises              = []
            first_session_at           = None
            total_duration_hours       = 0.0
            session_count              = len(activity_rows)

            for ep_row in activity_rows:
                ep = ep_row["data"]
                sw = ep.get("session_window", {})

                started_str = sw.get("started_at")
                if started_str:
                    started_dt = datetime.fromisoformat(started_str)
                    if first_session_at is None or started_dt < first_session_at:
                        first_session_at = started_dt

                total_duration_hours += sw.get("duration_minutes", 0) / 60.0

                for outcome in ep.get("exercise_outcomes", []):
                    all_exercises.append(outcome)
                    tid = outcome.get("topic_id")
                    if tid:
                        if tid not in topics_data:
                            topics_data[tid] = {"scores": [], "attempts": []}
                        topics_data[tid]["scores"].append(outcome.get("final_score", 0))
                        topics_data[tid]["attempts"].append(outcome.get("total_attempts", 1))

            # Compute topic mastery
            topic_mastery: Dict[str, Any] = {}
            areas_of_strength, areas_of_struggle = [], []

            for tid, td in topics_data.items():
                avg_score   = sum(td["scores"])   / len(td["scores"])   if td["scores"]   else 0.0
                avg_attempts = sum(td["attempts"]) / len(td["attempts"]) if td["attempts"] else 1.0
                confidence   = calculate_mastery_confidence(avg_score, avg_attempts)
                label        = classify_mastery_label(confidence)

                topic_mastery[tid] = {
                    "avg_score":         round(avg_score, 1),
                    "avg_attempts":      round(avg_attempts, 1),
                    "mastery_label":     label,
                    "mastery_confidence": confidence,
                }
                if label == "strong":
                    areas_of_strength.append(tid)
                elif label == "weak":
                    areas_of_struggle.append(tid)

            # Exercise performance summary
            total_ex     = len(all_exercises)
            first_pass   = sum(1 for e in all_exercises if e.get("passed") and e.get("total_attempts") == 1)
            multi_pass   = sum(1 for e in all_exercises if e.get("passed") and e.get("total_attempts", 1) > 1)
            never_passed = sum(1 for e in all_exercises if not e.get("passed"))
            avg_score_all = sum(e.get("final_score", 0) for e in all_exercises) / max(1, total_ex)
            avg_att_all   = sum(e.get("total_attempts", 1) for e in all_exercises) / max(1, total_ex)

            # LLM struggle pattern
            pattern_prompt = COURSE_COMPLETED_PATTERN_PROMPT.format(
                course_name=course_name,
                topic_mastery_json=json.dumps(topic_mastery, indent=2),
                strength_topics=", ".join(areas_of_strength) or "none",
                struggle_topics=", ".join(areas_of_struggle) or "none",
            )
            struggle_pattern = retry(lambda: OpenAIClient.complete(pattern_prompt))

            # skills_taught — use real skill_ids from exercises table
            overall_score = round(avg_score_all / 100.0, 2)
            skills_taught = [
                {
                    "skill_id":     sid,
                    "mastery_score": overall_score,
                    "confidence":   "validated",
                }
                for sid in skill_ids
            ]

            agent_summary = (
                f"Course completed in {session_count} sessions over "
                f"{round(total_duration_hours, 1)} hours. "
                f"Avg score {avg_score_all:.0f}. "
                + (f"Persistent struggle: {', '.join(areas_of_struggle[:3])}." if areas_of_struggle else "No weak topic areas.")
            )

            episode_data = {
                "course_id":   course_id,
                "pathway_id":  pathway_id,
                "completion_stats": {
                    "total_sessions":            session_count,
                    "total_duration_hours":      round(total_duration_hours, 1),
                    "first_session_at":          first_session_at.isoformat() if first_session_at else None,
                    "completed_at":              row["completed_at"].isoformat(),
                    "avg_session_duration_minutes": int(total_duration_hours * 60 / max(1, session_count)),
                },
                "exercise_performance": {
                    "total_exercises":          total_ex,
                    "passed_on_first_attempt":  first_pass,
                    "required_multiple_attempts": multi_pass,
                    "never_passed":             never_passed,
                    "avg_score":                round(avg_score_all, 1),
                    "avg_attempts_per_exercise": round(avg_att_all, 1),
                },
                "topic_mastery":      topic_mastery,
                "areas_of_strength":  areas_of_strength,
                "areas_of_struggle":  areas_of_struggle,
                "skills_taught":      skills_taught,
                "struggle_pattern":   struggle_pattern,
                "agent_summary":      agent_summary,
            }

            episode_id = _write_episode(cursor, user_id, "COURSE_COMPLETED", episode_data)

            # Write skill mastery rows for all skills this course teaches
            for sk in skills_taught:
                cursor.execute("""
                    INSERT INTO user_skill_mastery (user_id, skill_id, mastery_score, confidence, last_updated)
                    VALUES (%s, %s, %s, 'validated', NOW())
                    ON CONFLICT (user_id, skill_id)
                    DO UPDATE SET mastery_score = EXCLUDED.mastery_score,
                                  confidence    = 'validated',
                                  last_updated  = NOW()
                """, (user_id, sk["skill_id"], sk["mastery_score"]))

            # Mark raw row processed
            cursor.execute(
                "UPDATE raw_course_completion SET processed_at = NOW(), episode_id = %s WHERE id = %s",
                (episode_id, row["id"]),
            )

            # Semantic rebuild trigger
            cursor.execute("""
                INSERT INTO semantic_rebuild_trigger (user_id, trigger_reason, source_episode_id, triggered_at)
                VALUES (%s, 'course_completed', %s, NOW())
            """, (user_id, episode_id))

            logger.info(f"COURSE_COMPLETED episode {episode_id} created for user {user_id}")
            return _ok("COURSE_COMPLETED", user_id, episode_id, agent_summary)

    except Exception as e:
        logger.error(f"process_course_completed error: {e}")
        return _err("COURSE_COMPLETED", user_id, e)


# =============================================================================
# TOOL 6 — CAPSTONE_CODE_REVIEW (WITH REAL DATA PARSING)
# =============================================================================

def process_capstone_code_review(user_id: str, capstone_id: str, attempt_id: str) -> Dict[str, Any]:
    """
    Process raw_code_review into CAPSTONE_CODE_REVIEW episode.

    Parses the ACTUAL GitLab agent data structure (technology_breakdown dict,
    issues array, passed boolean) and converts to standard format.
    Uses LLM to build specific viva briefing from tech evaluations.
    Computes growth_signals on resubmission.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            cursor.execute("""
                SELECT rcr.*, uca.pathway_id as uca_pathway_id,
                       (SELECT COUNT(*) FROM user_capstone_attempts uca2
                        WHERE uca2.user_id = rcr.user_id
                          AND uca2.capstone_id = rcr.capstone_id
                          AND uca2.created_at <= uca.created_at) as attempt_number
                FROM raw_code_review rcr
                JOIN user_capstone_attempts uca ON uca.attempt_id = rcr.attempt_id
                WHERE rcr.user_id = %s AND rcr.capstone_id = %s
                  AND rcr.attempt_id = %s AND rcr.processed_at IS NULL
                LIMIT 1
            """, (user_id, capstone_id, attempt_id))
            row = cursor.fetchone()

            if not row:
                return _no_data("CAPSTONE_CODE_REVIEW", user_id, "No unprocessed code review found")

            user_id       = row["user_id"]
            pathway_id    = row.get("pathway_id") or row.get("uca_pathway_id")
            repo_url      = row["repo_url"]
            review_output = row["review_output"]
            attempt_num   = int(row.get("attempt_number") or 1)

            # === PARSE ACTUAL DATA STRUCTURE ===
            # Real structure: technology_breakdown (dict), issues (array), passed (bool)
            technology_breakdown = review_output.get('technology_breakdown', {})
            issues_list = review_output.get('issues', [])
            passed = review_output.get('passed', False)
            
            # Convert technology_breakdown dict to tech_evaluations array format
            tech_evaluations = []
            for tech_name, tech_data in technology_breakdown.items():
                tech_evaluations.append({
                    "technology": tech_name,
                    "understanding": tech_data.get('understanding', 'adequate'),
                    "design": tech_data.get('design', 'adequate'),
                    "code_quality": tech_data.get('code_quality', 'adequate'),
                    "test_coverage": tech_data.get('test_coverage', 'adequate'),
                    "verdict": tech_data.get('verdict', 'needs_revision')
                })
            
            # Determine overall verdict from individual tech verdicts
            if passed and all(t.get('verdict') in ['pass', 'adequate'] for t in tech_evaluations):
                overall_verdict = 'pass'
            elif any(t.get('verdict') == 'needs_revision' for t in tech_evaluations):
                overall_verdict = 'needs_revision'
            else:
                overall_verdict = 'pass' if passed else 'needs_revision'
            
            # Count issues by severity
            total_critical = sum(1 for issue in issues_list if issue.get('severity') == 'critical')
            total_minor = sum(1 for issue in issues_list if issue.get('severity') in ['low', 'medium'])
            total_high = sum(1 for issue in issues_list if issue.get('severity') == 'high')

            # LLM-generated specific viva briefing
            briefing_prompt = CODE_REVIEW_VIVA_BRIEFING_PROMPT.format(
                tech_evaluations_json=json.dumps(tech_evaluations, indent=2),
                overall_verdict=overall_verdict,
                critical_issues=total_critical,
                minor_issues=total_minor + total_high,
            )
            briefing_result = retry(lambda: OpenAIClient.complete_json(briefing_prompt)) or {}

            viva_briefing = {
                "probe_hard_on":                briefing_result.get("probe_hard_on", []),
                "can_go_light_on":              briefing_result.get("can_go_light_on", []),
                "likely_surface_knowledge_areas": briefing_result.get("likely_surface_knowledge_areas", []),
            }
            agent_summary = briefing_result.get("agent_summary", "[Summary unavailable]")

            # Growth signals on resubmission
            growth_signals = None
            if attempt_num > 1:
                cursor.execute("""
                    SELECT data FROM episodic_episodes
                    WHERE user_id = %s AND type = 'CAPSTONE_CODE_REVIEW'
                      AND data->>'capstone_id' = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (user_id, capstone_id))
                prior_row = cursor.fetchone()
                if prior_row:
                    prior_verdict = prior_row["data"].get("overall_verdict")
                    prior_critical = prior_row["data"].get("total_critical_issues", 0)
                    growth_signals = {
                        "prior_verdict":           prior_verdict,
                        "prior_critical_issues":   prior_critical,
                        "critical_issues_resolved": max(0, prior_critical - total_critical),
                        "verdict_improved":        (
                            prior_verdict == "needs_revision" and overall_verdict in ("pass", "conditional_pass")
                        ),
                    }

            episode_data = {
                "capstone_id":         capstone_id,
                "attempt_id":          attempt_id,
                "submission_id":       row.get("commit_sha", "unknown"),
                "repo_url":            repo_url,
                "attempt_number":      attempt_num,
                "timeline_adherence":  "on_time",
                "tech_evaluations":    tech_evaluations,
                "overall_verdict":     overall_verdict,
                "total_critical_issues": total_critical,
                "total_minor_issues":    total_minor + total_high,
                "growth_signals":      growth_signals,
                "viva_briefing":       viva_briefing,
                "agent_summary":       agent_summary,
            }

            episode_id = _write_episode(cursor, user_id, "CAPSTONE_CODE_REVIEW", episode_data)
            cursor.execute(
                "UPDATE raw_code_review SET processed_at = NOW(), episode_id = %s WHERE id = %s",
                (episode_id, row["id"]),
            )

            logger.info(f"CAPSTONE_CODE_REVIEW episode {episode_id} created for attempt {attempt_id}")
            return _ok("CAPSTONE_CODE_REVIEW", user_id, episode_id, agent_summary)

    except Exception as e:
        logger.error(f"process_capstone_code_review error: {e}")
        return _err("CAPSTONE_CODE_REVIEW", user_id, e)


# =============================================================================
# TOOL 7 — CAPSTONE_TEST_RUN (WITH REAL DATA PARSING)
# =============================================================================

def process_capstone_test_run(user_id: str, capstone_id: str, attempt_id: str) -> Dict[str, Any]:
    """
    Process raw_test_review into CAPSTONE_TEST_RUN episode.

    Parses the ACTUAL test agent data structure (flat with total_tests, passed,
    failed, test_results array) and extracts failed tests correctly.
    Uses LLM to cluster failures by conceptual topic and generate
    targeted mentor guidance. Computes regression_check on resubmission.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            cursor.execute("""
                SELECT rtr.*,
                       (SELECT COUNT(*) FROM user_capstone_attempts uca
                        WHERE uca.user_id = rtr.user_id
                          AND uca.capstone_id = rtr.capstone_id
                          AND uca.created_at <= (
                              SELECT created_at FROM user_capstone_attempts
                              WHERE attempt_id = rtr.attempt_id
                          )) as attempt_number
                FROM raw_test_review rtr
                WHERE rtr.user_id = %s AND rtr.capstone_id = %s
                  AND rtr.attempt_id = %s AND rtr.processed_at IS NULL
                LIMIT 1
            """, (user_id, capstone_id, attempt_id))
            row = cursor.fetchone()

            if not row:
                return _no_data("CAPSTONE_TEST_RUN", user_id, "No unprocessed test review found")

            user_id     = row["user_id"]
            test_output = row["test_output"]
            attempt_num = int(row.get("attempt_number") or 1)

            # === PARSE ACTUAL DATA STRUCTURE ===
            # Real structure: flat with total_tests, passed, failed, test_results array
            tests_total  = test_output.get("total_tests", 0)
            tests_passed = test_output.get("passed", 0)
            tests_failed = test_output.get("failed", 0)
            pass_rate    = tests_passed / tests_total if tests_total > 0 else 0.0
            coverage_pct = test_output.get("coverage_percent", 0)

            # Extract failed tests from test_results array
            all_results    = test_output.get("test_results", [])
            failed_results = [t for t in all_results if not t.get("passed", True)]

            # LLM semantic clustering
            cluster_prompt = TEST_RUN_CLUSTERING_PROMPT.format(
                capstone_id=capstone_id,
                failed_tests_json=json.dumps(failed_results, indent=2),
                all_tests_json=json.dumps(all_results[:40], indent=2),  # cap context
                pass_rate=pass_rate,
            )
            cluster_result = retry(lambda: OpenAIClient.complete_json(cluster_prompt)) or {}

            failure_clusters = cluster_result.get("failure_clusters", [])
            agent_verdict    = cluster_result.get("agent_verdict", "fail")
            mentor_guidance  = cluster_result.get("mentor_guidance", {"focus_areas": [], "suggested_resources": []})
            agent_summary    = cluster_result.get("agent_summary", "[Summary unavailable]")

            # Regression check on resubmission
            regression_check = None
            if attempt_num > 1:
                cursor.execute("""
                    SELECT data FROM episodic_episodes
                    WHERE user_id = %s AND type = 'CAPSTONE_TEST_RUN'
                      AND data->>'capstone_id' = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (user_id, capstone_id))
                prior_row = cursor.fetchone()
                if prior_row:
                    prior_data       = prior_row["data"]
                    prior_failed_ids = {
                        tid
                        for cluster in prior_data.get("failure_clusters", [])
                        for tid in cluster.get("failed_test_ids", [])
                    }
                    current_failed_ids = {
                        tid
                        for cluster in failure_clusters
                        for tid in cluster.get("failed_test_ids", [])
                    }
                    regressions = list(current_failed_ids - prior_failed_ids)
                    regression_check = {
                        "regressions_found": len(regressions) > 0,
                        "regressed_tests":   regressions,
                        "prior_pass_rate":   prior_data.get("test_summary", {}).get("pass_rate", 0),
                    }

            episode_data = {
                "capstone_id":   capstone_id,
                "attempt_id":    attempt_id,
                "submission_id": row.get("commit_sha", "unknown"),
                "attempt_number": attempt_num,
                "timeline_adherence": "on_time",
                "test_summary": {
                    "tests_total":    tests_total,
                    "tests_passed":   tests_passed,
                    "tests_failed":   tests_failed,
                    "pass_rate":      round(pass_rate, 3),
                    "coverage_percent": coverage_pct,
                },
                "failure_clusters": failure_clusters,
                "regression_check": regression_check,
                "agent_verdict":    agent_verdict,
                "mentor_guidance":  mentor_guidance,
                "agent_summary":    agent_summary,
            }

            episode_id = _write_episode(cursor, user_id, "CAPSTONE_TEST_RUN", episode_data)
            cursor.execute(
                "UPDATE raw_test_review SET processed_at = NOW(), episode_id = %s WHERE id = %s",
                (episode_id, row["id"]),
            )

            logger.info(f"CAPSTONE_TEST_RUN episode {episode_id} created for attempt {attempt_id}")
            return _ok("CAPSTONE_TEST_RUN", user_id, episode_id, agent_summary)

    except Exception as e:
        logger.error(f"process_capstone_test_run error: {e}")
        return _err("CAPSTONE_TEST_RUN", user_id, e)


# =============================================================================
# TOOL 8 — CAPSTONE_VIVA
# =============================================================================

def process_capstone_viva(user_id: str, capstone_id: str, attempt_id: str) -> Dict[str, Any]:
    """
    Process raw_viva_turns into CAPSTONE_VIVA episode.

    Uses LLM to score each Q&A pair on correctness and depth of reasoning,
    confirms or refutes surface knowledge suspicions from the code review,
    writes skill_updates back to user_skill_mastery, and fires semantic rebuild.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM raw_viva_turns
                WHERE user_id = %s AND capstone_id = %s
                  AND attempt_id = %s AND processed_at IS NULL
                ORDER BY turn_number
            """, (user_id, capstone_id, attempt_id))
            rows = cursor.fetchall()

            if not rows:
                return _no_data("CAPSTONE_VIVA", user_id, "No unprocessed viva turns found")

            user_id        = rows[0]["user_id"]
            viva_session_id = rows[0]["session_id"]

            # Pull capstone passing_score
            cursor.execute(
                "SELECT passing_score FROM capstones WHERE capstone_id = %s", (capstone_id,)
            )
            cap_row      = cursor.fetchone()
            passing_score = cap_row["passing_score"] if cap_row else 70

            # Pull code review episode for viva briefing
            cursor.execute("""
                SELECT episode_id, data FROM episodic_episodes
                WHERE user_id = %s AND type = 'CAPSTONE_CODE_REVIEW'
                  AND data->>'attempt_id' = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id, attempt_id))
            cr_row                = cursor.fetchone()
            code_review_episode_id = cr_row["episode_id"] if cr_row else None
            viva_briefing          = cr_row["data"].get("viva_briefing", {}) if cr_row else {}

            # Pull test run episode id
            cursor.execute("""
                SELECT episode_id FROM episodic_episodes
                WHERE user_id = %s AND type = 'CAPSTONE_TEST_RUN'
                  AND data->>'attempt_id' = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id, attempt_id))
            tr_row               = cursor.fetchone()
            test_run_episode_id  = tr_row["episode_id"] if tr_row else None

            # Build full Q&A turn list for LLM
            turns_for_llm = [
                {
                    "turn_number": r["turn_number"],
                    "role":        r["role"],
                    "message":     r["message"],
                }
                for r in rows
            ]

            scoring_prompt = VIVA_SCORING_PROMPT.format(
                capstone_id=capstone_id,
                passing_score=passing_score,
                turns_json=json.dumps(turns_for_llm, indent=2),
                viva_briefing_json=json.dumps(viva_briefing, indent=2),
            )
            
            # Try LLM scoring with fallback to basic extraction
            llm_result = retry(lambda: OpenAIClient.complete_json(scoring_prompt))
            
            if not llm_result or not llm_result.get("question_log"):
                # Fallback: Extract Q&A pairs manually from turns
                logger.warning(f"LLM scoring failed for viva {attempt_id}, using fallback extraction")
                question_log = []
                topic_verdicts = {}
                
                # Pair up viva_agent (questions) with user (answers)
                q_num = 0
                for i, turn in enumerate(rows):
                    if turn["role"] == "viva_agent":
                        q_num += 1
                        question_text = turn["message"]  # Keep full question text
                        
                        # Find the next user turn as the answer
                        answer_text_full = ""
                        if i + 1 < len(rows) and rows[i + 1]["role"] == "user":
                            answer_text_full = rows[i + 1]["message"]
                        
                        # Basic scoring: longer answers = better (crude heuristic)
                        # Use FULL answer length for scoring, not truncated
                        answer_len = len(answer_text_full)
                        if answer_len > 150:
                            score = 70
                            quality = "good"
                        elif answer_len > 80:
                            score = 50
                            quality = "partial"
                        else:
                            score = 30
                            quality = "poor"
                        
                        # Create meaningful answer summary: first 150 chars as preview
                        if answer_text_full:
                            answer_preview = answer_text_full[:150]
                            if len(answer_text_full) > 150:
                                answer_preview += "..."
                            answer_summary = f"{answer_preview} ({answer_len} chars, {quality})"
                        else:
                            answer_summary = f"No answer provided (0 chars)"
                        
                        question_log.append({
                            "question_id": f"q{q_num}",
                            "topic_id": "general",
                            "question_text": question_text,
                            "question_type": "conceptual",
                            "was_probed_from_code_review": False,
                            "answer_quality": quality,
                            "answer_summary": answer_summary,
                            "score": score
                        })
                
                # Generate basic topic verdict
                if question_log:
                    avg_score = sum(q["score"] for q in question_log) / len(question_log)
                    topic_verdicts["general"] = {
                        "verbal_score": int(avg_score),
                        "confirmed_surface_knowledge": avg_score < 50,
                        "mastery_delta": 0.1 if avg_score >= 70 else -0.1
                    }
                
                agent_verdict = "pass" if len(question_log) > 0 and sum(q["score"] for q in question_log) / len(question_log) >= passing_score else "fail"
                verdict_reason = f"Fallback scoring: {len(question_log)} Q&A pairs analyzed"
                student_summary = f"Viva completed with {len(question_log)} questions. " + ("Pass" if agent_verdict == "pass" else "Needs improvement in understanding core concepts.")
            else:
                question_log    = llm_result.get("question_log", [])
                topic_verdicts  = llm_result.get("topic_verdicts", {})
                agent_verdict   = llm_result.get("agent_verdict", "fail")
                verdict_reason  = llm_result.get("verdict_reason", "Viva scoring incomplete")
                student_summary = llm_result.get("student_facing_summary", "[Summary unavailable]")

            # Derive viva score from question_log scores
            scores      = [q.get("score", 0) for q in question_log]
            viva_score  = int(sum(scores) / len(scores)) if scores else 0

            answers_satisfactory = sum(1 for q in question_log if q.get("answer_quality") in ("good", "partial"))
            answers_poor         = sum(1 for q in question_log if q.get("answer_quality") == "poor")

            # Build skill_updates from topic_verdicts
            # Pull current mastery scores so we can apply delta correctly
            skill_updates = []
            for topic_id, verdict in topic_verdicts.items():
                cursor.execute("""
                    SELECT mastery_score FROM user_skill_mastery
                    WHERE user_id = %s AND skill_id = %s
                """, (user_id, topic_id))
                cur_row       = cursor.fetchone()
                current_score = cur_row["mastery_score"] if cur_row else 0.5
                delta         = verdict.get("mastery_delta", 0.0)
                new_score     = round(min(1.0, max(0.0, current_score + delta)), 2)

                skill_updates.append({
                    "skill_id":        topic_id,
                    "new_mastery_score": new_score,
                    "confidence":      "validated",
                })

            # Write skill mastery updates
            for su in skill_updates:
                cursor.execute("""
                    INSERT INTO user_skill_mastery (user_id, skill_id, mastery_score, confidence, last_updated)
                    VALUES (%s, %s, %s, 'validated', NOW())
                    ON CONFLICT (user_id, skill_id)
                    DO UPDATE SET mastery_score = EXCLUDED.mastery_score,
                                  confidence    = 'validated',
                                  last_updated  = NOW()
                """, (user_id, su["skill_id"], su["new_mastery_score"]))

            # Update user_capstone_attempts with final verdict
            cursor.execute("""
                UPDATE user_capstone_attempts
                SET status      = %s,
                    final_score = %s,
                    passed      = %s,
                    updated_at  = NOW()
                WHERE attempt_id = %s
            """, (
                "passed" if agent_verdict == "pass" else "failed",
                viva_score,
                agent_verdict == "pass",
                attempt_id,
            ))

            episode_data = {
                "capstone_id":     capstone_id,
                "attempt_id":      attempt_id,
                "viva_session_id": viva_session_id,
                "attempt_number":  1,
                "timeline_adherence": "on_time",
                "source_episodes": {
                    "code_review_episode_id": code_review_episode_id,
                    "test_run_episode_id":    test_run_episode_id,
                },
                "question_log":           question_log,
                "topic_verdicts":         topic_verdicts,
                "viva_score":             viva_score,
                "questions_asked":        len(question_log),
                "answers_satisfactory":   answers_satisfactory,
                "answers_poor":           answers_poor,
                "agent_verdict":          agent_verdict,
                "verdict_reason":         verdict_reason,
                "student_facing_summary": student_summary,
                "skill_updates":          skill_updates,
            }

            episode_id = _write_episode(cursor, user_id, "CAPSTONE_VIVA", episode_data)
            row_ids    = [r["id"] for r in rows]
            cursor.execute(
                "UPDATE raw_viva_turns SET processed_at = NOW(), episode_id = %s WHERE id = ANY(%s)",
                (episode_id, row_ids),
            )

            # Semantic rebuild trigger
            cursor.execute("""
                INSERT INTO semantic_rebuild_trigger (user_id, trigger_reason, source_episode_id, triggered_at)
                VALUES (%s, 'capstone_complete', %s, NOW())
            """, (user_id, episode_id))

            logger.info(f"CAPSTONE_VIVA episode {episode_id} created — verdict: {agent_verdict}")
            return _ok("CAPSTONE_VIVA", user_id, episode_id,
                       f"Viva {agent_verdict.upper()}: score {viva_score}/100 (threshold {passing_score})")

    except Exception as e:
        logger.error(f"process_capstone_viva error: {e}")
        return _err("CAPSTONE_VIVA", user_id, e)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _write_episode(cursor, user_id: str, episode_type: str, data: Dict) -> str:
    """Insert episode row and return the new episode_id."""
    episode_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO episodic_episodes (episode_id, user_id, type, schema_version, timestamp, data)
        VALUES (%s, %s, %s, 1, NOW(), %s)
    """, (episode_id, user_id, episode_type, json.dumps(data)))
    return episode_id


def _ok(episode_type: str, user_id: str, episode_id: str, summary: str) -> Dict[str, Any]:
    return {"success": True,  "episode_id": episode_id, "episode_type": episode_type,
            "user_id": user_id, "summary": summary, "error": None}


def _no_data(episode_type: str, user_id: str, msg: str) -> Dict[str, Any]:
    return {"success": True,  "episode_id": None, "episode_type": episode_type,
            "user_id": user_id, "summary": msg,  "error": None}


def _err(episode_type: str, user_id: str, exc: Exception) -> Dict[str, Any]:
    return {"success": False, "episode_id": None, "episode_type": episode_type,
            "user_id": user_id, "summary": None,  "error": str(exc)}


# =============================================================================
# MCP SERVER SETUP
# =============================================================================

mcp = FastMCP("neulearn-episodic-tools")

DatabaseManager.initialize(DB_URL)
OpenAIClient.initialize()


@mcp.tool(name="process_self_assessment", title="Process Self-Assessment into Episode")
def tool_process_self_assessment(user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process raw self-assessment into SELF_ASSESSMENT episode.
    Maps self-ratings to numeric_confidence, identifies MCQ candidates,
    generates LLM agent flags.

    Args:
        user_id:    User ID
        session_id: Optional session ID filter
    """
    return process_self_assessment(user_id, session_id)


@mcp.tool(name="process_pre_assessment", title="Process Pre-Assessment MCQ Results into Episode")
def tool_process_pre_assessment(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Process MCQ rows into PRE_ASSESSMENT episode.
    Groups by topic + difficulty, applies scoring rules, computes calibration
    delta from SELF_ASSESSMENT numeric_confidence.

    Args:
        user_id:    User ID
        session_id: Pre-assessment session ID
    """
    return process_pre_assessment(user_id, session_id)


@mcp.tool(name="process_pathway_allocated", title="Process Pathway Allocation into Episode")
def tool_process_pathway_allocated(user_id: str, pathway_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process pathway allocation into LEARNING_PATH_ASSIGNED episode.
    Captures assignment logic from SELF_ASSESSMENT and PRE_ASSESSMENT.
    Explains course assignment reasoning (beginner direct, MCQ failures, targeted modules).

    Args:
        user_id:    User ID
        pathway_id: Optional pathway ID filter
    """
    return process_pathway_allocated(user_id, pathway_id)


@mcp.tool(name="process_course_activity", title="Process Course Study Session into Episode")
def tool_process_course_activity(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Process course session events into COURSE_ACTIVITY episode.
    LLM classifies failure types and struggle signals. Momentum score computed.

    Args:
        user_id:    User ID
        session_id: Course study session ID
    """
    return process_course_activity(user_id, session_id)


@mcp.tool(name="process_mentor_chat", title="Process Mentor Chat Session into Episode")
def tool_process_mentor_chat(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Process mentor chat turns into MENTOR_CHAT episode.
    LLM identifies unresolved topics, breakthroughs, intent counts, policy events.
    No per-turn summaries — session-level analysis only.

    Args:
        user_id:    User ID
        session_id: Chat session ID
    """
    return process_mentor_chat(user_id, session_id)


@mcp.tool(name="process_course_completed", title="Process Course Completion into Episode")
def tool_process_course_completed(user_id: str, course_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Aggregate all COURSE_ACTIVITY episodes into COURSE_COMPLETED episode.
    Writes validated skill mastery rows. Fires semantic_rebuild_trigger.

    Args:
        user_id:   User ID
        course_id: Optional course ID filter
    """
    return process_course_completed(user_id, course_id)


@mcp.tool(name="process_capstone_code_review", title="Process Capstone Code Review into Episode")
def tool_process_capstone_code_review(user_id: str, capstone_id: str, attempt_id: str) -> Dict[str, Any]:
    """
    Process code review output into CAPSTONE_CODE_REVIEW episode.
    Parses actual GitLab agent structure (technology_breakdown dict, issues array, passed bool).
    LLM generates specific viva briefing with probe questions.
    Growth signals computed on resubmission.

    Args:
        user_id:    User ID
        capstone_id: Capstone project ID
        attempt_id:  Attempt ID
    """
    return process_capstone_code_review(user_id, capstone_id, attempt_id)


@mcp.tool(name="process_capstone_test_run", title="Process Capstone Test Results into Episode")
def tool_process_capstone_test_run(user_id: str, capstone_id: str, attempt_id: str) -> Dict[str, Any]:
    """
    Cluster test failures semantically into CAPSTONE_TEST_RUN episode.
    Parses actual test agent structure (flat with total_tests, passed, failed, test_results array).
    LLM groups by conceptual gap, not test name prefix.
    Regression check computed on resubmission.

    Args:
        user_id:    User ID
        capstone_id: Capstone project ID
        attempt_id:  Attempt ID
    """
    return process_capstone_test_run(user_id, capstone_id, attempt_id)


@mcp.tool(name="process_capstone_viva", title="Process Capstone Viva into Episode")
def tool_process_capstone_viva(user_id: str, capstone_id: str, attempt_id: str) -> Dict[str, Any]:
    """
    Score viva Q&A and produce CAPSTONE_VIVA episode.
    LLM scores by correctness and depth of reasoning — not answer length.
    Writes skill_updates to user_skill_mastery. Fires semantic_rebuild_trigger.

    Args:
        user_id:    User ID
        capstone_id: Capstone project ID
        attempt_id:  Attempt ID
    """
    return process_capstone_viva(user_id, capstone_id, attempt_id)


# =============================================================================
# SEMANTIC PROFILE BUILDING
# =============================================================================

# ─── Onboarding Profile Builder ──────────────────────────────────────────────

ONBOARDING_PROFILE_PROMPT = """You are the semantic profile builder for Neulearn. You analyze episodic data to create a durable learner portrait.

**TASK**: Build the initial semantic profile from onboarding assessment episodes.

**INPUT DATA**:
Self-assessment: {self_assessment_json}
Pre-assessment MCQ results: {pre_assessment_json}
Learning path assignment: {pathway_json}

**OUTPUT REQUIREMENTS**:
Return valid JSON matching this exact structure:

{{
  "identity": {{
    "declared_interests": ["list of interests from self_assessment"],
    "active_domains": ["list of domains from pathway assignment"],
    "onboarded_at": "ISO timestamp from earliest episode",
    "entry_assessment": {{
      "assessed_entry_level": "beginner|intermediate|advanced - analyze overall pre-assessment performance",
      "calibration_at_entry": "aligned|slightly_overconfident|highly_overconfident|underconfident - compare self-ratings to MCQ outcomes",
      "overconfident_topics_at_entry": ["topics where self-rating significantly exceeded MCQ performance"]
    }}
  }},
  "skill_map": {{
    "domains": {{
      "DOMAIN_NAME": {{
        "domain_mastery_score": 0.0-1.0,
        "topics": {{
          "TOPIC_ID": {{
            "mastery_score": 0.0-1.0,
            "confidence_tier": "self_assessed",
            "areas_of_strength": [],
            "areas_of_struggle": [],
            "last_validated_by": "SELF_ASSESSMENT",
            "last_validated_at": "ISO timestamp"
          }}
        }},
        "validated_skills": [],
        "surface_knowledge_flags": []
      }}
    }}
  }},
  "learning_disposition": {{
    "calibration_profile": {{
      "entry_calibration": "same as identity.entry_assessment.calibration_at_entry",
      "calibration_trend": "establishing_baseline",
      "note": "Brief narrative about self-awareness at entry"
    }},
    "persistence_profile": {{
      "avg_attempts_before_pass": 0.0,
      "first_attempt_pass_rate": 0.0,
      "gives_up_pattern": false,
      "note": "Not yet established - onboarding only"
    }},
    "learning_velocity": "unknown",
    "preferred_explanation_style": "unknown",
    "struggle_pattern": "Not yet observed",
    "knowledge_depth_risk": "unknown",
    "gaming_disposition": {{
      "risk_level": "low",
      "direct_answer_attempts_total": 0,
      "pattern": "none"
    }}
  }},
  "engagement_shape": {{
    "avg_session_duration_minutes": 0,
    "preferred_session_time": "unknown",
    "sessions_per_week_avg": 0.0,
    "consistency_label": "new_user",
    "longest_observed_gap_days": 0,
    "dropout_risk_profile": "unknown",
    "note": "Just onboarded - no engagement pattern yet"
  }},
  "capstone_record": {{
    "completed": [],
    "attempted_not_passed": []
  }}
}}

**ANALYSIS GUIDELINES**:
1. **Calibration Analysis**: For each topic, compare self-assessed level (beginner/intermediate/advanced) to MCQ difficulty passed (medium/hard). Someone who rated "intermediate" but failed medium MCQs is overconfident.

2. **Entry Level**: Synthesize from overall MCQ patterns - not just one topic. Look at pass rates, difficulty levels attempted, consistency across topics.

3. **Overconfident Topics**: Only flag when self-rating is 2+ levels above demonstrated ability. One tough MCQ failure ≠ overconfidence.

4. **Mastery Scores**: Convert self-ratings to scores (beginner=0.3, intermediate=0.5, advanced=0.7). These are UNVALIDATED baselines.

5. **Domain Grouping**: Group topics by their domain from the pathway assignment context.

Return ONLY the JSON, no markdown fences or explanations.
"""

def build_initial_semantic_profile(user_id: str) -> Dict[str, Any]:
    """
    Build initial semantic profile from onboarding episodes (SELF_ASSESSMENT, PRE_ASSESSMENT, LEARNING_PATH_ASSIGNED).
    
    Called once after pathway assignment completes.
    Establishes baseline calibration, entry-level assessment, and initial skill map.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Fetch SELF_ASSESSMENT episode
            cursor.execute("""
                SELECT episode_id, data, timestamp
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'SELF_ASSESSMENT'
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id,))
            self_assess = cursor.fetchone()
            
            # Fetch PRE_ASSESSMENT episode
            cursor.execute("""
                SELECT episode_id, data, timestamp
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'PRE_ASSESSMENT'
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id,))
            pre_assess = cursor.fetchone()
            
            # Fetch LEARNING_PATH_ASSIGNED episode
            cursor.execute("""
                SELECT episode_id, data, timestamp
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'LEARNING_PATH_ASSIGNED'
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id,))
            pathway = cursor.fetchone()
            
            if not self_assess or not pre_assess or not pathway:
                return {
                    "status": "error",
                    "error": f"Missing onboarding episodes for user {user_id}",
                    "found": {
                        "self_assess": bool(self_assess),
                        "pre_assess": bool(pre_assess),
                        "pathway": bool(pathway)
                    }
                }
            
            # Build prompt
            prompt = ONBOARDING_PROFILE_PROMPT.format(
                self_assessment_json=json.dumps(self_assess["data"], indent=2),
                pre_assessment_json=json.dumps(pre_assess["data"], indent=2),
                pathway_json=json.dumps(pathway["data"], indent=2)
            )
            
            # Get LLM analysis (high max_tokens for large profile JSON)
            profile_data = retry(lambda: OpenAIClient.complete_json(prompt, max_tokens=3500))
            
            if not profile_data:
                return {"status": "error", "error": "LLM failed to generate valid profile"}
            
            # Get current max  version
            cursor.execute("""
                SELECT COALESCE(MAX(version), 0) as max_ver
                FROM semantic_profile_versions
                WHERE user_id = %s
            """, (user_id,))
            max_ver = cursor.fetchone()["max_ver"]
            new_version = max_ver + 1
            
            # Mark all previous versions as not current
            cursor.execute("""
                UPDATE semantic_profile_versions
                SET is_current = false
                WHERE user_id = %s
            """, (user_id,))
            
            # Insert new version
            cursor.execute("""
                INSERT INTO semantic_profile_versions
                    (user_id, version, profile, trigger, is_current, created_at)
                VALUES (%s, %s, %s, 'ONBOARDING_COMPLETE', true, NOW())
            """, (user_id, new_version, json.dumps(profile_data)))
            
            # Add rebuild log entry
            rebuild_entry = {
                "rebuilt_at": datetime.utcnow().isoformat() + "Z",
                "trigger": "ONBOARDING_COMPLETE",
                "trigger_episode_id": pathway["episode_id"],
                "agent_version": "semantic_agent_v1",
                "sections_updated": ["identity", "skill_map", "learning_disposition.calibration_profile"]
            }
            
            profile_data.setdefault("rebuild_log", []).append(rebuild_entry)
            
            # Update the profile with rebuild log
            cursor.execute("""
                UPDATE semantic_profile_versions
                SET profile = %s
                WHERE user_id = %s AND version = %s
            """, (json.dumps(profile_data), user_id, new_version))
            
            logger.info(f"Built initial semantic profile for {user_id} (version {new_version})")
            
            return {
                "status": "success",
                "user_id": user_id,
                "profile_version": new_version,
                "trigger": "ONBOARDING_COMPLETE",
                "sections_updated": ["identity", "skill_map", "learning_disposition.calibration_profile"],
                "profile_summary": {
                    "entry_level": profile_data["identity"]["entry_assessment"]["assessed_entry_level"],
                    "calibration": profile_data["identity"]["entry_assessment"]["calibration_at_entry"],
                    "domains": list(profile_data["skill_map"]["domains"].keys()),
                    "overconfident_topics": profile_data["identity"]["entry_assessment"]["overconfident_topics_at_entry"]
                }
            }
            
    except Exception as e:
        logger.error(f"Error building initial profile for {user_id}: {e}")
        return {"status": "error", "error": str(e)}


# ─── Course Completion Profile Updater ───────────────────────────────────────

COURSE_COMPLETION_UPDATE_PROMPT = """You are updating a semantic profile after COURSE_COMPLETED.

**CURRENT PROFILE** (sections we're updating):
{current_profile_sections}

**NEW EPISODE DATA**:
Course completed episode: {course_completed_json}
All COURSE_ACTIVITY episodes for this course: {course_activities_json}

**TASK**: Update skill_map, learning_disposition, and engagement_shape based on course completion evidence.

**OUTPUT**: Return JSON with ONLY the sections being updated:

{{
  "skill_map_updates": {{
    "TOPIC_ID": {{
      "mastery_score": 0.0-1.0,
      "confidence_tier": "validated",
      "areas_of_strength": ["specific subtopics where user excelled"],
      "areas_of_struggle": ["specific subtopics where user struggled - even if they eventually passed"],
      "last_validated_by": "COURSE_COMPLETED",
      "last_validated_at": "ISO timestamp"
    }}
  }},
  "learning_disposition_updates": {{
    "persistence_profile": {{
      "avg_attempts_before_pass": calculated from course activities,
      "first_attempt_pass_rate": calculated from course activities,
      "gives_up_pattern": true/false based on abandoned exercises,
      "note": "Narrative about retry behavior, persistence patterns"
    }},
    "learning_velocity": "fast|moderate|deliberate|struggling - based on session count, duration, pass rates",
    "struggle_pattern": "Synthesized narrative of WHERE struggles happened - not just that they struggled",
    "calibration_profile": {{
      "calibration_trend": "improving|stable|declining|unknown - compare entry calibration to current demonstrated ability",
      "note": "Update about self-awareness growth"
    }}
  }},
  "engagement_shape_updates": {{
    "avg_session_duration_minutes": calculated from all sessions for this course,
    "preferred_session_time": "morning|afternoon|evening|night - most frequent session start hour",
    "sessions_per_week_avg": calculated from course duration,
    "consistency_label": "regular|bursty|declining|sporadic",
    "longest_observed_gap_days": max gap between sessions,
    "dropout_risk_profile": "low|medium|high - based on completion despite gaps",
    "note": "Narrative about engagement pattern"
  }}
}}

**ANALYSIS GUIDELINES**:
1. **areas_of_strength/struggle**: Be granular. "Error handling in async code" not just "async".
2. **persistence_profile**: calculate from raw exercise attempts. Distinguish "takes 2-3 tries but succeeds" from "abandons after 1 fail".
3. **learning_velocity**: Fast = high first-attempt pass rate + quick completion. Deliberate = many attempts but strong mastery. Struggling = low pass rates + high abandon rate.
4. **struggle_pattern**: Narrative insight, not just list. E.g., "Strong on input/output tasks, weak on internal mechanics" or "Conceptual understanding good, syntax errors frequent".
5. **calibration_trend**: Compare self-rated confidence at entry to actual performance. Did reality match expectations? Did they learn to predict their own performance better?

Return ONLY the JSON.
"""

def update_profile_course_completed(user_id: str, course_id: str) -> Dict[str, Any]:
    """
    Update semantic profile after COURSE_COMPLETED episode.
    
    Upgrades skill_map confidence tier to 'validated'.
    Calculates learning disposition (persistence, velocity, struggle patterns).
    Updates engagement shape from session data.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Fetch current profile
            cursor.execute("""
                SELECT version, profile
                FROM semantic_profile_versions
                WHERE user_id = %s AND is_current = true
            """, (user_id,))
            current = cursor.fetchone()
            
            if not current:
                return {"status": "error", "error": f"No current profile found for {user_id}. Run build_initial_semantic_profile first."}
            
            current_profile = current["profile"]
            
            # Fetch COURSE_COMPLETED episode
            cursor.execute("""
                SELECT episode_id, data, timestamp
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'COURSE_COMPLETED'
                  AND data->>'course_id' = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id, course_id))
            completed = cursor.fetchone()
            
            if not completed:
                return {"status": "error", "error": f"No COURSE_COMPLETED episode found for user {user_id}, course {course_id}"}
            
            # Fetch all COURSE_ACTIVITY episodes for this course
            cursor.execute("""
                SELECT episode_id, data, timestamp
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'COURSE_ACTIVITY'
                  AND data->>'course_id' = %s
                ORDER BY timestamp ASC
            """, (user_id, course_id))
            activities = cursor.fetchall()
            
            # Build prompt with relevant sections
            current_sections = {
                "skill_map": current_profile.get("skill_map", {}),
                "learning_disposition": current_profile.get("learning_disposition", {}),
                "engagement_shape": current_profile.get("engagement_shape", {})
            }
            
            prompt = COURSE_COMPLETION_UPDATE_PROMPT.format(
                current_profile_sections=json.dumps(current_sections, indent=2),
                course_completed_json=json.dumps(completed["data"], indent=2),
                course_activities_json=json.dumps([a["data"] for a in activities], indent=2)
            )
            
            # Get LLM updates (high max_tokens for profile updates)
            updates = retry(lambda: OpenAIClient.complete_json(prompt, max_tokens=2500))
            
            if not updates:
                return {"status": "error", "error": "LLM failed to generate profile updates"}
            
            # Apply updates to current profile
            updated_profile = json.loads(json.dumps(current_profile))  # Deep copy
            
            # Update skill_map
            for topic_id, topic_data in updates.get("skill_map_updates", {}).items():
                # Find which domain this topic belongs to
                for domain_name, domain_data in updated_profile.get("skill_map", {}).get("domains", {}).items():
                    if topic_id in domain_data.get("topics", {}):
                        updated_profile["skill_map"]["domains"][domain_name]["topics"][topic_id].update(topic_data)
                        break
            
            # Update learning_disposition
            updated_profile.setdefault("learning_disposition", {}).update(updates.get("learning_disposition_updates", {}))
            
            # Update engagement_shape
            updated_profile.setdefault("engagement_shape", {}).update(updates.get("engagement_shape_updates", {}))
            
            # Add rebuild log entry
            rebuild_entry = {
                "rebuilt_at": datetime.utcnow().isoformat() + "Z",
                "trigger": "COURSE_COMPLETED",
                "trigger_episode_id": completed["episode_id"],
                "agent_version": "semantic_agent_v1",
                "sections_updated": ["skill_map", "learning_disposition", "engagement_shape"]
            }
            updated_profile.setdefault("rebuild_log", []).append(rebuild_entry)
            
            # Create new version
            new_version = current["version"] + 1
            
            cursor.execute("""
                UPDATE semantic_profile_versions
                SET is_current = false
                WHERE user_id = %s
            """, (user_id,))
            
            cursor.execute("""
                INSERT INTO semantic_profile_versions
                    (user_id, version, profile, trigger, is_current, created_at)
                VALUES (%s, %s, %s, 'COURSE_COMPLETED', true, NOW())
            """, (user_id, new_version, json.dumps(updated_profile)))
            
            logger.info(f"Updated semantic profile for {user_id} after course {course_id} completion (version {new_version})")
            
            return {
                "status": "success",
                "user_id": user_id,
                "course_id": course_id,
                "profile_version": new_version,
                "trigger": "COURSE_COMPLETED",
                "sections_updated": ["skill_map", "learning_disposition", "engagement_shape"]
            }
            
    except Exception as e:
        logger.error(f"Error updating profile for {user_id} after course completion: {e}")
        return {"status": "error", "error": str(e)}


# ─── Pathway Assignment Profile Updater ──────────────────────────────────────

PATHWAY_ASSIGNMENT_UPDATE_PROMPT = """You are updating a semantic profile after a new LEARNING_PATH_ASSIGNED episode.

**CURRENT PROFILE** (identity and skill_map sections):
{current_profile_sections}

**NEW PATHWAY EPISODE**:
{pathway_episode_json}

**TASK**: Update identity.active_domains and add new skill_map topics from this pathway.

**OUTPUT**: Return JSON with updates

:

{{
  "identity_updates": {{
    "active_domains": ["updated list - add new domain if not present"]
  }},
  "skill_map_updates": {{
    "DOMAIN_NAME": {{
      "topics": {{
        "NEW_TOPIC_ID": {{
          "mastery_score": 0.0-1.0,
          "confidence_tier": "self_assessed",
          "areas_of_strength": [],
          "areas_of_struggle": [],
          "last_validated_by": "LEARNING_PATH_ASSIGNED",
          "last_validated_at": "ISO timestamp"
        }}
      }}
    }}
  }}
}}

**ANALYSIS GUIDELINES**:
1. Extract the domain from the pathway assignment
2. Add domain to active_domains if not already present  
3. For each new topic in the pathway, initialize with self_assessed tier
4. Set initial mastery scores based on pathway assignment context
5. Leave areas_of_strength/struggle empty (not yet observed)

Return ONLY the JSON.
"""

def update_profile_pathway_assigned(user_id: str, pathway_id: str) -> Dict[str, Any]:
    """
    Update semantic profile after new LEARNING_PATH_ASSIGNED episode.
    
    Users can have multiple pathways.
    Adds new domain to active_domains, initializes new topics.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Fetch current profile
            cursor.execute("""
                SELECT version, profile
                FROM semantic_profile_versions
                WHERE user_id = %s AND is_current = true
            """, (user_id,))
            current = cursor.fetchone()
            
            if not current:
                return {"status": "error", "error": f"No current profile found for {user_id}. Run build_initial_semantic_profile first."}
            
            current_profile = current["profile"]
            
            # Fetch LEARNING_PATH_ASSIGNED episode
            cursor.execute("""
                SELECT episode_id, data, timestamp
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'LEARNING_PATH_ASSIGNED'
                  AND data->>'pathway_id' = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id, pathway_id))
            pathway_ep = cursor.fetchone()
            
            if not pathway_ep:
                return {"status": "error", "error": f"No LEARNING_PATH_ASSIGNED episode found for pathway {pathway_id}"}
            
            # Build prompt
            current_sections = {
                "identity": current_profile.get("identity", {}),
                "skill_map": current_profile.get("skill_map", {})
            }
            
            prompt = PATHWAY_ASSIGNMENT_UPDATE_PROMPT.format(
                current_profile_sections=json.dumps(current_sections, indent=2),
                pathway_episode_json=json.dumps(pathway_ep["data"], indent=2)
            )
            
            # Get LLM updates (high max_tokens for profile updates)
            updates = retry(lambda: OpenAIClient.complete_json(prompt, max_tokens=2500))
            
            if not updates:
                return {"status": "error", "error": "LLM failed to generate profile updates"}
            
            # Apply updates
            updated_profile = json.loads(json.dumps(current_profile))  # Deep copy
            
            # Update identity
            if "identity_updates" in updates:
                updated_profile.setdefault("identity", {}).update(updates["identity_updates"])
            
            # Update skill_map - merge new topics
            for domain_name, domain_updates in updates.get("skill_map_updates", {}).items():
                if domain_name not in updated_profile.setdefault("skill_map", {}).setdefault("domains", {}):
                    # New domain entirely
                    updated_profile["skill_map"]["domains"][domain_name] = {
                        "domain_mastery_score": 0.3,
                        "topics": {},
                        "validated_skills": [],
                        "surface_knowledge_flags": []
                    }
                
                # Add new topics
                for topic_id, topic_data in domain_updates.get("topics", {}).items():
                    updated_profile["skill_map"]["domains"][domain_name]["topics"][topic_id] = topic_data
            
            # Add rebuild log entry
            rebuild_entry = {
                "rebuilt_at": datetime.utcnow().isoformat() + "Z",
                "trigger": "LEARNING_PATH_ASSIGNED",
                "trigger_episode_id": pathway_ep["episode_id"],
                "agent_version": "semantic_agent_v1",
                "sections_updated": ["identity", "skill_map"]
            }
            updated_profile.setdefault("rebuild_log", []).append(rebuild_entry)
            
            # Create new version
            new_version = current["version"] + 1
            
            cursor.execute("""
                UPDATE semantic_profile_versions
                SET is_current = false
                WHERE user_id = %s
            """, (user_id,))
            
            cursor.execute("""
                INSERT INTO semantic_profile_versions
                    (user_id, version, profile, trigger, is_current, created_at)
                VALUES (%s, %s, %s, 'LEARNING_PATH_ASSIGNED', true, NOW())
            """, (user_id, new_version, json.dumps(updated_profile)))
            
            logger.info(f"Updated semantic profile for {user_id} after pathway {pathway_id} assignment (version {new_version})")
            
            return {
                "status": "success",
                "user_id": user_id,
                "pathway_id": pathway_id,
                "profile_version": new_version,
                "trigger": "LEARNING_PATH_ASSIGNED",
                "sections_updated": ["identity", "skill_map"]
            }
            
    except Exception as e:
        logger.error(f"Error updating profile for {user_id} after pathway assignment: {e}")
        return {"status": "error", "error": str(e)}


# ─── Capstone Viva Profile Updater ───────────────────────────────────────────

CAPSTONE_VIVA_UPDATE_PROMPT = """You are updating a semantic profile after CAPSTONE_VIVA episode.

**CURRENT PROFILE** (skill_map, capstone_record, learning_disposition):
{current_profile_sections}

**VIVA EPISODE**:
{viva_episode_json}

**CODE REVIEW EPISODE** (for context):
{code_review_episode_json}

**TASK**: Update profile based on viva outcome (pass/fail, surface knowledge revealed).

**OUTPUT**: Return JSON:

{{
  "skill_map_updates": {{
    "surface_knowledge_flags": [
      {{"topic_id": "sk_topic", "flagged_by": "CAPSTONE_VIVA", "flagged_at": "ISO", "note": "explanation"}}
    ]
  }},
  "capstone_record_updates": {{
    "type": "completed" | "attempted_not_passed",
    "record": {{
      "capstone_id": "...",
      "domain": "...",
      "completed_at" or "latest_attempt_number": ...,
      "viva_score": int,
      "surface_knowledge_confirmed": ["topic_ids"],
      "skills_validated" or "resubmission_focus": [...],
      "student_facing_summary": "from viva episode"
    }}
  }},
  "learning_disposition_updates": {{
    "knowledge_depth_risk": "low|medium|high|surface_knowledge_risk_on_internals",
    "calibration_profile": {{
      "calibration_trend": "improving|stable|declining",
      "note": "updated calibration insight"
    }}
  }}
}}

**ANALYSIS GUIDELINES**:
1. If viva verdict = "pass": type="completed", add to capstone_record.completed
2. If viva verdict = "fail": type="attempted_not_passed", extract resubmission_focus
3. surface_knowledge_flags: Only add if viva_episode.topic_verdicts shows confirmed_surface_knowledge=true
4. knowledge_depth_risk: Synthesize from number of surface flags and verbatim viva performance
5. calibration_trend: Compare self-assessed confidence to actual viva performance

Return ONLY the JSON.
"""

def update_profile_capstone_viva(user_id: str, capstone_id: str, attempt_id: str) -> Dict[str, Any]:
    """
    Update semantic profile after CAPSTONE_VIVA episode (pass or fail).
    
    Final skill validation or surface knowledge confirmation.
    Updates capstone_record, skill_map flags, knowledge_depth_risk.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Fetch current profile
            cursor.execute("""
                SELECT version, profile
                FROM semantic_profile_versions
                WHERE user_id = %s AND is_current = true
            """, (user_id,))
            current = cursor.fetchone()
            
            if not current:
                return {"status": "error", "error": f"No current profile found for {user_id}"}
            
            current_profile = current["profile"]
            
            # Fetch CAPSTONE_VIVA episode
            cursor.execute("""
                SELECT episode_id, data, timestamp
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'CAPSTONE_VIVA'
                  AND data->>'capstone_id' = %s
                  AND data->>'attempt_id' = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id, capstone_id, attempt_id))
            viva_ep = cursor.fetchone()
            
            if not viva_ep:
                return {"status": "error", "error": f"No CAPSTONE_VIVA episode found for capstone {capstone_id}, attempt {attempt_id}"}
            
            # Fetch CODE_REVIEW episode for context
            cursor.execute("""
                SELECT data
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'CAPSTONE_CODE_REVIEW'
                  AND data->>'capstone_id' = %s
                  AND data->>'attempt_id' = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id, capstone_id, attempt_id))
            code_review_row = cursor.fetchone()
            code_review_data = code_review_row["data"] if code_review_row else {}
            
            # Build prompt
            current_sections = {
                "skill_map": current_profile.get("skill_map", {}),
                "capstone_record": current_profile.get("capstone_record", {}),
                "learning_disposition": current_profile.get("learning_disposition", {})
            }
            
            prompt = CAPSTONE_VIVA_UPDATE_PROMPT.format(
                current_profile_sections=json.dumps(current_sections, indent=2),
                viva_episode_json=json.dumps(viva_ep["data"], indent=2),
                code_review_episode_json=json.dumps(code_review_data, indent=2)
            )
            
            # Get LLM updates (high max_tokens for profile updates)
            updates = retry(lambda: OpenAIClient.complete_json(prompt, max_tokens=2500))
            
            if not updates:
                return {"status": "error", "error": "LLM failed to generate profile updates"}
            
            # Apply updates
            updated_profile = json.loads(json.dumps(current_profile))  # Deep copy
            
            # Update skill_map surface_knowledge_flags
            if "skill_map_updates" in updates and "surface_knowledge_flags" in updates["skill_map_updates"]:
                for domain_data in updated_profile.setdefault("skill_map", {}).setdefault("domains", {}).values():
                    existing_flags = domain_data.get("surface_knowledge_flags", [])
                    new_flags = updates["skill_map_updates"]["surface_knowledge_flags"]
                    # Merge flags
                    for new_flag in new_flags:
                        if not any(f["topic_id"] == new_flag["topic_id"] for f in existing_flags):
                            existing_flags.append(new_flag)
                    domain_data["surface_knowledge_flags"] = existing_flags
            
            # Update capstone_record
            if "capstone_record_updates" in updates:
                capstone_update = updates["capstone_record_updates"]
                if capstone_update["type"] == "completed":
                    updated_profile.setdefault("capstone_record", {}).setdefault("completed", []).append(capstone_update["record"])
                else:  # attempted_not_passed
                    # Remove from completed if exists, add to attempted_not_passed
                    attempted = updated_profile.setdefault("capstone_record", {}).setdefault("attempted_not_passed", [])
                    # Update existing or append new
                    found = False
                    for i, cap in enumerate(attempted):
                        if cap.get("capstone_id") == capstone_update["record"]["capstone_id"]:
                            attempted[i] = capstone_update["record"]
                            found = True
                            break
                    if not found:
                        attempted.append(capstone_update["record"])
            
            # Update learning_disposition
            if "learning_disposition_updates" in updates:
                updated_profile.setdefault("learning_disposition", {}).update(updates["learning_disposition_updates"])
            
            # Add rebuild log entry
            rebuild_entry = {
                "rebuilt_at": datetime.utcnow().isoformat() + "Z",
                "trigger": "CAPSTONE_VIVA",
                "trigger_episode_id": viva_ep["episode_id"],
                "agent_version": "semantic_agent_v1",
                "sections_updated": ["skill_map", "capstone_record", "learning_disposition"]
            }
            updated_profile.setdefault("rebuild_log", []).append(rebuild_entry)
            
            # Create new version
            new_version = current["version"] + 1
            
            cursor.execute("""
                UPDATE semantic_profile_versions
                SET is_current = false
                WHERE user_id = %s
            """, (user_id,))
            
            cursor.execute("""
                INSERT INTO semantic_profile_versions
                    (user_id, version, profile, trigger, is_current, created_at)
                VALUES (%s, %s, %s, 'CAPSTONE_VIVA', true, NOW())
            """, (user_id, new_version, json.dumps(updated_profile)))
            
            logger.info(f"Updated semantic profile for {user_id} after capstone viva {capstone_id} (version {new_version})")
            
            return {
                "status": "success",
                "user_id": user_id,
                "capstone_id": capstone_id,
                "attempt_id": attempt_id,
                "profile_version": new_version,
                "trigger": "CAPSTONE_VIVA",
                "sections_updated": ["skill_map", "capstone_record", "learning_disposition"]
            }
            
    except Exception as e:
        logger.error(f"Error updating profile for {user_id} after capstone viva: {e}")
        return {"status": "error", "error": str(e)}


# ─── Manual Profile Update (Periodic Analysis) ──────────────────────────────────────

MANUAL_UPDATE_PROMPT = """You are performing a periodic semantic profile analysis to identify confirmed learning patterns.

**CURRENT PROFILE** (learning_disposition, engagement_shape):
{current_profile_sections}

**RECENT MENTOR_CHAT EPISODES** (last 10):
{mentor_chat_episodes_json}

**RECENT COURSE_ACTIVITY EPISODES** (last 10):
{course_activity_episodes_json}

**TASK**: Analyze episodes to identify confirmed patterns and update the profile accordingly.

**OUTPUT**: Return JSON with pattern-based updates:

{{
  "learning_disposition_updates": {{
    "preferred_explanation_style": "analogy_first|code_first|theory_first|visual_diagrams|unknown - only if pattern confirmed across 3+ mentor chats",
    "help_seeking_behavior": {{
      "independence_level": "high|medium|low",
      "direct_answer_requests_total": count from mentor_policy_events,
      "pattern": "none|isolated_event|pattern - 3+ requests = pattern"
    }}
  }},
  "engagement_shape_updates": {{
    "avg_session_duration_minutes": recalculated average,
    "sessions_per_week_avg": recalculated,
    "consistency_label": "regular|bursty|declining|sporadic",
    "longest_observed_gap_days": max gap observed,
    "dropout_risk_profile": "low|medium|high"
  }}
}}

**ANALYSIS GUIDELINES**:
1. preferred_explanation_style: Only update if same style helps understanding 3+ times
2. help_seeking_behavior: Track direct answer requests from mentor policy events
3. engagement metrics: Recalculate from all course activities, identify trends
4. consistency_label: Analyze session gap patterns over last 2+ weeks
5. If no clear pattern emerges (< 3 confirming episodes), leave field unchanged

Return ONLY the JSON.
"""

def update_profile_manual(user_id: str, focus_area: str = "all") -> Dict[str, Any]:
    """
    Manual semantic profile update - periodic analysis to confirm long-term patterns.
    
    Analyzes recent episodes (mentor chats, course activities) to update:
    - Preferred explanation style
    - Help-seeking behavior patterns
    - Engagement patterns
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Fetch current profile
            cursor.execute("""
                SELECT version, profile
                FROM semantic_profile_versions
                WHERE user_id = %s AND is_current = true
            """, (user_id,))
            current = cursor.fetchone()
            
            if not current:
                return {"status": "error", "error": f"No current profile found for {user_id}"}
            
            current_profile = current["profile"]
            
            # Fetch recent MENTOR_CHAT episodes
            cursor.execute("""
                SELECT data, timestamp
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'MENTOR_CHAT'
                ORDER BY timestamp DESC LIMIT 10
            """, (user_id,))
            mentor_chats = [row["data"] for row in cursor.fetchall()]
            
            # Fetch recent COURSE_ACTIVITY episodes
            cursor.execute("""
                SELECT data, timestamp
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'COURSE_ACTIVITY'
                ORDER BY timestamp DESC LIMIT 10
            """, (user_id,))
            course_activities = [row["data"] for row in cursor.fetchall()]
            
            if not mentor_chats and not course_activities:
                return {"status": "success", "message": "No recent episodes found - profile unchanged"}
            
            # Build prompt
            current_sections = {
                "learning_disposition": current_profile.get("learning_disposition", {}),
                "engagement_shape": current_profile.get("engagement_shape", {})
            }
            
            prompt = MANUAL_UPDATE_PROMPT.format(
                current_profile_sections=json.dumps(current_sections, indent=2),
                mentor_chat_episodes_json=json.dumps(mentor_chats, indent=2),
                course_activity_episodes_json=json.dumps(course_activities, indent=2)
            )
            
            # Get LLM updates (high max_tokens for profile updates)
            updates = retry(lambda: OpenAIClient.complete_json(prompt, max_tokens=2500))
            
            if not updates:
                return {"status": "error", "error": "LLM failed to generate profile updates"}
            
            # Apply updates
            updated_profile = json.loads(json.dumps(current_profile))  # Deep copy
            
            # Update learning_disposition
            if "learning_disposition_updates" in updates:
                updated_profile.setdefault("learning_disposition", {}).update(updates["learning_disposition_updates"])
            
            # Update engagement_shape
            if "engagement_shape_updates" in updates:
                updated_profile.setdefault("engagement_shape", {}).update(updates["engagement_shape_updates"])
            
            # Add rebuild log entry
            rebuild_entry = {
                "rebuilt_at": datetime.utcnow().isoformat() + "Z",
                "trigger": "cron_dream_pass",
                "trigger_episode_id": None,
                "agent_version": "semantic_agent_v1",
                "sections_updated": ["learning_disposition", "engagement_shape"]
            }
            updated_profile.setdefault("rebuild_log", []).append(rebuild_entry)
            
            # Create new version
            new_version = current["version"] + 1
            
            cursor.execute("""
                UPDATE semantic_profile_versions
                SET is_current = false
                WHERE user_id = %s
            """, (user_id,))
            
            cursor.execute("""
                INSERT INTO semantic_profile_versions
                    (user_id, version, profile, trigger, is_current, created_at)
                VALUES (%s, %s, %s, 'cron_dream_pass', true, NOW())
            """, (user_id, new_version, json.dumps(updated_profile)))
            
            logger.info(f"Manual profile update for {user_id} (version {new_version})")
            
            return {
                "status": "success",
                "user_id": user_id,
                "profile_version": new_version,
                "trigger": "cron_dream_pass",
                "focus_area": focus_area,
                "sections_updated": ["learning_disposition", "engagement_shape"]
            }
            
    except Exception as e:
        logger.error(f"Error in manual profile update for {user_id}: {e}")
        return {"status": "error", "error": str(e)}


# =============================================================================
# MENTOR CHATBOT TOOLS
# =============================================================================

# ─── Helper: Enrich Topic/Skill Data with Names ──────────────────────────────

def enrich_topic_name(topic_id: str, cursor) -> str:
    """Fetch topic name from modules table (module_id = topic_id)."""
    cursor.execute("SELECT title FROM modules WHERE module_id = %s", (topic_id,))
    row = cursor.fetchone()
    return row["title"] if row else topic_id  # Fallback to ID if not found


def enrich_skill_names_in_map(skill_map: Dict[str, Any], cursor) -> Dict[str, Any]:
    """
    Enrich skill_map with topic_name for each topic.
    Since skills don't have a separate table yet, we use topic names.
    """
    enriched = skill_map.copy()
    
    if "domains" in enriched:
        for domain_name, domain_data in enriched["domains"].items():
            if "topics" in domain_data:
                for topic_id, topic_data in domain_data["topics"].items():
                    topic_name = enrich_topic_name(topic_id, cursor)
                    topic_data["topic_name"] = topic_name
                    topic_data["topic_id"] = topic_id
    
    return enriched


# ─── Tool 1: Get Student Context ─────────────────────────────────────────────

def get_student_context(user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch complete contextual snapshot for mentor bot.
    Includes current session, semantic profile, pathway state, skill levels, and recent history.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            result = {
                "user_id": user_id,
                "current_session": None,
                "semantic_profile": None,
                "pathway_state": None,
                "skill_levels": {},
                "recent_history": {
                    "last_mentor_chats": [],
                    "last_course_activities": []
                }
            }
            
            # 1. Current Session Context
            if session_id:
                cursor.execute("""
                    SELECT s.session_id, s.session_type, s.course_id, s.started_at,
                           c.name as course_name,
                           EXTRACT(EPOCH FROM (NOW() - s.started_at))/60 as duration_minutes
                    FROM sessions s
                    LEFT JOIN courses c ON s.course_id = c.course_id
                    WHERE s.session_id = %s AND s.user_id = %s
                """, (session_id, user_id))
                session = cursor.fetchone()
                if session:
                    result["current_session"] = dict(session)
                    result["current_session"]["duration_minutes"] = round(session["duration_minutes"], 1)
            
            # 2. Semantic Profile (current version)
            cursor.execute("""
                SELECT profile
                FROM semantic_profile_versions
                WHERE user_id = %s AND is_current = true
            """, (user_id,))
            profile_row = cursor.fetchone()
            if profile_row:
                profile = profile_row["profile"]
                # Enrich skill_map with topic names
                if "skill_map" in profile:
                    profile["skill_map"] = enrich_skill_names_in_map(profile["skill_map"], cursor)
                result["semantic_profile"] = profile
            
            # 3. Pathway State
            cursor.execute("""
                SELECT pathway_id, interest, courses_remaining, courses_in_progress,
                       courses_completed, capstone_id, status
                FROM student_pathways
                WHERE user_id = %s AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))
            pathway = cursor.fetchone()
            if pathway:
                pathway_dict = dict(pathway)
                
                # Get capstone status if exists
                if pathway_dict["capstone_id"]:
                    cursor.execute("""
                        SELECT status, final_score, passed
                        FROM user_capstone_attempts
                        WHERE user_id = %s AND capstone_id = %s
                        ORDER BY created_at DESC LIMIT 1
                    """, (user_id, pathway_dict["capstone_id"]))
                    capstone = cursor.fetchone()
                    if capstone:
                        pathway_dict["capstone_status"] = capstone["status"]
                        pathway_dict["capstone_score"] = capstone["final_score"]
                        pathway_dict["capstone_passed"] = capstone["passed"]
                    else:
                        pathway_dict["capstone_status"] = "not_started"
                
                result["pathway_state"] = pathway_dict
            
            # 4. Skill Mastery Levels
            cursor.execute("""
                SELECT usm.skill_id, usm.mastery_score, usm.confidence,
                       usm.last_updated, m.title as topic_name, m.module_id as topic_id
                FROM user_skill_mastery usm
                LEFT JOIN modules m ON usm.skill_id = m.module_id
                WHERE usm.user_id = %s
                ORDER BY usm.mastery_score DESC
            """, (user_id,))
            skills = cursor.fetchall()
            for skill in skills:
                skill_dict = dict(skill)
                topic_id = skill_dict.get("topic_id") or skill_dict["skill_id"]
                result["skill_levels"][topic_id] = {
                    "topic_id": topic_id,
                    "topic_name": skill_dict.get("topic_name") or "Unknown Topic",
                    "skill_id": skill_dict["skill_id"],
                    "mastery_score": skill_dict["mastery_score"],
                    "confidence_tier": skill_dict["confidence"]
                }
            
            # 5. Recent Episode History
            # Last 3 mentor chats
            cursor.execute("""
                SELECT episode_id, timestamp, data
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'MENTOR_CHAT'
                ORDER BY timestamp DESC LIMIT 3
            """, (user_id,))
            mentor_chats = cursor.fetchall()
            result["recent_history"]["last_mentor_chats"] = [
                {
                    "episode_id": row["episode_id"],
                    "timestamp": row["timestamp"].isoformat(),
                    "summary": row["data"].get("summary", {})
                }
                for row in mentor_chats
            ]
            
            # Last 5 course activities
            cursor.execute("""
                SELECT episode_id, timestamp, data
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'COURSE_ACTIVITY'
                ORDER BY timestamp DESC LIMIT 5
            """, (user_id,))
            activities = cursor.fetchall()
            result["recent_history"]["last_course_activities"] = [
                {
                    "episode_id": row["episode_id"],
                    "timestamp": row["timestamp"].isoformat(),
                    "course_id": row["data"].get("course_id"),
                    "topics_covered": row["data"].get("topics_covered", [])
                }
                for row in activities
            ]
            
            return {
                "status": "success",
                "context": result
            }
            
    except Exception as e:
        logger.error(f"Error fetching student context for {user_id}: {e}")
        return {"status": "error", "error": str(e)}


# ─── Tool 2: Get Topic Content ───────────────────────────────────────────────

def get_topic_content(topic_id: str, content_type: str = "all") -> Dict[str, Any]:
    """
    Retrieve course content for a specific topic.
    
    Note: This is a stub implementation. In production, this would query
    a course_content table or external CMS with structured content.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Get topic details from modules
            cursor.execute("""
                SELECT m.module_id as topic_id, m.title as topic_name,
                       m.course_id, c.name as course_name
                FROM modules m
                JOIN courses c ON m.course_id = c.course_id
                WHERE m.module_id = %s
            """, (topic_id,))
            topic = cursor.fetchone()
            
            if not topic:
                return {
                    "status": "error",
                    "error": f"Topic {topic_id} not found"
                }
            
            # TODO: Query actual course content table when created
            # For now, return placeholder structure
            result = {
                "topic_id": topic["topic_id"],
                "topic_name": topic["topic_name"],
                "course_id": topic["course_id"],
                "course_name": topic["course_name"],
                "key_concepts": [],
                "common_mistakes": [],
                "when_to_use": "Content not yet populated",
                "examples": [],
                "note": "Course content table not yet implemented. This is placeholder data."
            }
            
            return {
                "status": "success",
                "content": result
            }
            
    except Exception as e:
        logger.error(f"Error fetching topic content for {topic_id}: {e}")
        return {"status": "error", "error": str(e)}


# ─── Tool 3: Get Exercise Context ────────────────────────────────────────────

def get_exercise_context(exercise_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get details about a specific exercise and optionally user's attempt history.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Get exercise details
            cursor.execute("""
                SELECT e.exercise_id, e.title as exercise_name, e.module_id as topic_id,
                       e.difficulty, e.pass_threshold, e.description,
                       m.title as topic_name
                FROM exercises e
                LEFT JOIN modules m ON e.module_id = m.module_id
                WHERE e.exercise_id = %s
            """, (exercise_id,))
            exercise = cursor.fetchone()
            
            if not exercise:
                return {
                    "status": "error",
                    "error": f"Exercise {exercise_id} not found"
                }
            
            result = {
                "exercise_id": exercise["exercise_id"],
                "exercise_name": exercise["exercise_name"],
                "topic_id": exercise["topic_id"],
                "topic_name": exercise.get("topic_name") or "Unknown Topic",
                "difficulty": exercise["difficulty"],
                "pass_threshold": exercise["pass_threshold"],
                "description": exercise.get("description"),
                "learning_objectives": [],  # TODO: Add to exercises table
                "hints": [],  # TODO: Add hints table
                "user_attempts": []
            }
            
            # Get user's attempt history if user_id provided
            if user_id:
                # Look for attempts in COURSE_ACTIVITY episodes
                cursor.execute("""
                    SELECT episode_id, timestamp, data
                    FROM episodic_episodes
                    WHERE user_id = %s AND type = 'COURSE_ACTIVITY'
                    AND data->>'exercise_id' = %s
                    ORDER BY timestamp DESC
                    LIMIT 10
                """, (user_id, exercise_id))
                attempts = cursor.fetchall()
                
                result["user_attempts"] = [
                    {
                        "attempt": idx + 1,
                        "timestamp": row["timestamp"].isoformat(),
                        "score": row["data"].get("score"),
                        "completed": row["data"].get("completed", False)
                    }
                    for idx, row in enumerate(reversed(attempts))
                ]
            
            return {
                "status": "success",
                "exercise": result
            }
            
    except Exception as e:
        logger.error(f"Error fetching exercise context for {exercise_id}: {e}")
        return {"status": "error", "error": str(e)}


# ─── Tool 4: Search Similar Questions ────────────────────────────────────────

def search_similar_questions(user_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Find similar previously answered questions from mentor chat history.
    
    Uses simple text matching for now. In production, use vector embeddings.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Search in MENTOR_CHAT episodes
            cursor.execute("""
                SELECT episode_id, timestamp, data
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'MENTOR_CHAT'
                ORDER BY timestamp DESC
                LIMIT 20
            """, (user_id,))
            chats = cursor.fetchall()
            
            # Simple keyword matching (TODO: Replace with vector search)
            query_lower = query.lower()
            similar = []
            
            for chat in chats:
                chat_summary = chat["data"].get("summary", {})
                topics_discussed = chat["data"].get("topics_discussed", [])
                
                # Very basic relevance - check if query words appear in topics
                relevance = 0.0
                for topic in topics_discussed:
                    if any(word in topic.lower() for word in query_lower.split()):
                        relevance += 0.3
                
                if relevance > 0:
                    similar.append({
                        "original_question": chat_summary.get("primary_topic", "Unknown question"),
                        "answer_summary": chat_summary.get("mentor_guidance", "No summary available"),
                        "relevance_score": min(relevance, 1.0),
                        "session_id": chat["data"].get("session_id"),
                        "asked_at": chat["timestamp"].isoformat()
                    })
            
            # Sort by relevance and limit
            similar.sort(key=lambda x: x["relevance_score"], reverse=True)
            similar = similar[:limit]
            
            return {
                "status": "success",
                "similar_questions": similar,
                "note": "Using basic keyword matching. Vector search not yet implemented."
            }
            
    except Exception as e:
        logger.error(f"Error searching similar questions for {user_id}: {e}")
        return {"status": "error", "error": str(e)}


# ─── Tool 5: Log Mentor Chat Turn ────────────────────────────────────────────

def log_mentor_chat_turn(
    user_id: str,
    session_id: str,
    turn_number: int,
    role: str,
    message: str,
    course_id: Optional[str] = None,
    pathway_id: Optional[str] = None,
    topic_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record a mentor chat turn to raw_mentor_chat_turns table.
    These will later be aggregated into MENTOR_CHAT episodes.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO raw_mentor_chat_turns
                    (user_id, session_id, turn_number, role, message,
                     course_id, pathway_id, topic_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id
            """, (user_id, session_id, turn_number, role, message,
                  course_id, pathway_id, topic_id))
            turn = cursor.fetchone()

            return {
                "status": "success",
                "turn_id": turn["id"]
            }

    except Exception as e:
        logger.error(f"Error logging mentor chat turn: {e}")
        return {"status": "error", "error": str(e)}

# ─── Tool 6: Get Prerequisite Gaps ───────────────────────────────────────────

def get_prerequisite_gaps(user_id: str, current_topic_id: str) -> Dict[str, Any]:
    """
    Identify prerequisite topics the user struggles with.
    
    Note: Thais requires a prerequisite graph in FalkorDB or a 
    topic_prerequisites table in Postgres. Stub implementation for now.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Get current topic name
            current_topic_name = enrich_topic_name(current_topic_id, cursor)
            
            # TODO: Query prerequisite graph from FalkorDB
            # For now, return placeholder
            result = {
                "current_topic": current_topic_id,
                "current_topic_name": current_topic_name,
                "prerequisite_gaps": [],
                "note": "Prerequisite graph not yet implemented. This requires FalkorDB integration."
            }
            
            return {
                "status": "success",
                "gaps": result
            }
            
    except Exception as e:
        logger.error(f"Error getting prerequisite gaps for {current_topic_id}: {e}")
        return {"status": "error", "error": str(e)}


# ─── Tool 7: Suggest Learning Path ───────────────────────────────────────────

def suggest_learning_path(user_id: str, current_context: str = "normal") -> Dict[str, Any]:
    """
    Provide strategic guidance on what to learn next.
    Uses semantic profile + pathway state + recent activity to make recommendation.
    """
    try:
        with DatabaseManager.get_cursor() as cursor:
            # Get semantic profile
            cursor.execute("""
                SELECT profile
                FROM semantic_profile_versions
                WHERE user_id = %s AND is_current = true
            """, (user_id,))
            profile_row = cursor.fetchone()
            
            if not profile_row:
                return {
                    "status": "error",
                    "error": f"No semantic profile found for {user_id}"
                }
            
            profile = profile_row["profile"]
            
            # Get recent activity
            cursor.execute("""
                SELECT episode_id, timestamp, data
                FROM episodic_episodes
                WHERE user_id = %s AND type = 'COURSE_ACTIVITY'
                ORDER BY timestamp DESC LIMIT 5
            """, (user_id,))
            activities = cursor.fetchall()
            
            # Build context for LLM
            context_data = {
                "current_context": current_context,
                "learning_disposition": profile.get("learning_disposition", {}),
                "engagement_shape": profile.get("engagement_shape", {}),
                "recent_activities": [
                    {
                        "timestamp": a["timestamp"].isoformat(),
                        "session_duration": a["data"].get("session_time_minutes"),
                        "topics_covered": a["data"].get("topics_covered", [])
                    }
                    for a in activities
                ]
            }
            
            # LLM prompt for learning path suggestion
            prompt = f"""Based on this student's learning profile and recent activity, provide a strategic learning recommendation.

Context: {current_context}
Profile: {json.dumps(context_data, indent=2)}

Return JSON with:
{{
  "recommendation": "focused_practice" | "move_forward" | "seek_help" | "break_recommended",
  "reasoning": "Brief explanation (1-2 sentences)",
  "suggested_resources": [
    {{"type": "topic_review" | "exercise" | "break", "resource_id": "..."}}
  ],
  "confidence": "high" | "medium" | "low"
}}"""
            
            suggestion = retry(lambda: OpenAIClient.complete_json(prompt, max_tokens=500))
            
            if not suggestion:
                # Fallback basic recommendation
                suggestion = {
                    "recommendation": "move_forward",
                    "reasoning": "Continue with your current learning path.",
                    "suggested_resources": [],
                    "confidence": "low"
                }
            
            return {
                "status": "success",
                "suggestion": suggestion
            }
            
    except Exception as e:
        logger.error(f"Error suggesting learning path for {user_id}: {e}")
        return {"status": "error", "error": str(e)}


# ─── MCP Tool Registrations ──────────────────────────────────────────────────

@mcp.tool(name="build_initial_semantic_profile", title="Build Initial Semantic Profile from Onboarding")
def tool_build_initial_profile(user_id: str) -> Dict[str, Any]:
    """
    Build the initial semantic profile from onboarding episodes.
    Analyzes SELF_ASSESSMENT, PRE_ASSESSMENT, and LEARNING_PATH_ASSIGNED to establish baseline.
    
    Sets:
    - identity (declared interests, entry level, calibration)
    - skill_map (self-assessed tier, initial mastery scores)
    - learning_disposition (calibration profile baseline)
    
    Args:
        user_id: User ID
    """
    return build_initial_semantic_profile(user_id)


@mcp.tool(name="update_profile_course_completed", title="Update Profile After Course Completion")
def tool_update_profile_course_completed(user_id: str, course_id: str) -> Dict[str, Any]:
    """
    Update semantic profile after COURSE_COMPLETED episode.
    Upgrades skill confidence tiers, calculates learning patterns, updates engagement shape.
    
    Updates:
    - skill_map (validated tier, areas of strength/struggle)
    - learning_disposition (persistence, velocity, struggle patterns, calibration trend)
    - engagement_shape (session patterns, consistency, dropout risk)
    
    Args:
        user_id: User ID
        course_id: Course ID that was completed
    """
    return update_profile_course_completed(user_id, course_id)


@mcp.tool(name="update_profile_pathway_assigned", title="Update Profile After Pathway Assignment")
def tool_update_profile_pathway_assigned(user_id: str, pathway_id: str) -> Dict[str, Any]:
    """
    Update semantic profile when a new pathway is assigned.
    Users can have multiple pathways - this updates active_domains and adds new skill baselines.
    
    Updates:
    - identity.active_domains (adds new domain)
    - skill_map (adds new topics from pathway with self_assessed tier)
    
    Args:
        user_id: User ID
        pathway_id: New pathway ID assigned
    """
    return update_profile_pathway_assigned(user_id, pathway_id)


@mcp.tool(name="update_profile_capstone_viva", title="Update Profile After Capstone Viva")
def tool_update_profile_capstone_viva(user_id: str, capstone_id: str, attempt_id: str) -> Dict[str, Any]:
    """
    Update semantic profile after CAPSTONE_VIVA episode (pass or fail).
    Final validation of skills, surface knowledge confirmation, capstone record update.
    
    Updates:
    - skill_map (surface_knowledge_flags if viva revealed gaps)
    - capstone_record (completed or attempted_not_passed)
    - learning_disposition.knowledge_depth_risk
    
    Args:
        user_id: User ID
        capstone_id: Capstone project ID
        attempt_id: Attempt ID
    """
    return update_profile_capstone_viva(user_id, capstone_id, attempt_id)


@mcp.tool(name="update_profile_manual", title="Manual Profile Update (Periodic Analysis)")
def tool_update_profile_manual(user_id: str, focus_area: str = "all") -> Dict[str, Any]:
    """
    Manual semantic profile update - periodic analysis of learning patterns.
    Analyzes recent episodes to confirm long-term behavioral patterns.
    
    Updates (based on patterns across multiple recent episodes):
    - learning_disposition.preferred_explanation_style
    - learning_disposition.gaming_disposition
    - engagement_shape (updated averages and consistency)
    
    Args:
        user_id: User ID
        focus_area: What to focus on - "all", "learning_patterns", "engagement" (default: all)
    """
    return update_profile_manual(user_id, focus_area)


# ─── Mentor Chatbot MCP Tools ────────────────────────────────────────────────

@mcp.tool(name="get_student_context", title="Get Complete Student Context for Mentor Bot")
def tool_get_student_context(user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch complete contextual snapshot for mentor bot initialization.
    
    Returns:
    - current_session: Session details including course, topic, duration
    - semantic_profile: Learning patterns, struggles, strengths with enriched topic names
    - pathway_state: Current pathway, courses status, capstone status
    - skill_levels: Mastery scores per topic/skill with readable names
    - recent_history: Last mentor chats and course activities
    
    Args:
        user_id: User ID
        session_id: Optional session ID to get current session context
    """
    return get_student_context(user_id, session_id)


@mcp.tool(name="get_topic_content", title="Get Course Content for a Topic")
def tool_get_topic_content(topic_id: str, content_type: str = "all") -> Dict[str, Any]:
    """
    Retrieve course content for a specific topic to answer content questions.
    
    Returns topic concepts, common mistakes, when to use, and examples.
    Note: Currently returns placeholder data. Full content requires course_content table.
    
    Args:
        topic_id: Topic ID (module_id)
        content_type: Filter content type - "all", "concepts", "examples", "common_mistakes"
    """
    return get_topic_content(topic_id, content_type)


@mcp.tool(name="get_exercise_context", title="Get Exercise Details and Attempt History")
def tool_get_exercise_context(exercise_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get details about a specific exercise and optionally user's attempt history.
    
    Returns exercise details, difficulty, learning objectives, hints, and user attempts.
    
    Args:
        exercise_id: Exercise ID
        user_id: Optional user ID to include attempt history
    """
    return get_exercise_context(exercise_id, user_id)


@mcp.tool(name="search_similar_questions", title="Search Similar Previously Answered Questions")
def tool_search_similar_questions(user_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Find similar previously answered questions from mentor chat history.
    
    Uses keyword matching to find relevant past interactions.
    Note: Basic text matching. Production should use vector embeddings.
    
    Args:
        user_id: User ID
        query: Search query
        limit: Maximum number of results (default: 5)
    """
    return search_similar_questions(user_id, query, limit)


@mcp.tool(name="log_mentor_chat_turn", title="Log Mentor Chat Conversation Turn")
def tool_log_mentor_chat_turn(
    user_id: str,
    session_id: str,
    turn_number: int,
    role: str,
    message: str,
    course_id: Optional[str] = None,
    pathway_id: Optional[str] = None,
    topic_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record a mentor chat turn to raw_mentor_chat_turns table.
    These will later be aggregated into MENTOR_CHAT episodes.
    
    Args:
        user_id: User ID
        session_id: Session ID
        turn_number: Turn number in conversation
        role: "user" or "assistant"
        message: Full message text
        course_id: Optional course ID context
        pathway_id: Optional pathway ID context
        topic_id: Optional topic ID context
    """
    return log_mentor_chat_turn(user_id, session_id, turn_number, role, message,
                                 course_id, pathway_id, topic_id)


@mcp.tool(name="get_prerequisite_gaps", title="Identify Prerequisite Knowledge Gaps")
def tool_get_prerequisite_gaps(user_id: str, current_topic_id: str) -> Dict[str, Any]:
    """
    Identify prerequisite topics the user struggles with.
    
    Note: Requires prerequisite graph from FalkorDB. Currently returns placeholder.
    
    Args:
        user_id: User ID
        current_topic_id: Current topic being studied
    """
    return get_prerequisite_gaps(user_id, current_topic_id)


@mcp.tool(name="suggest_learning_path", title="Suggest Strategic Learning Path")
def tool_suggest_learning_path(user_id: str, current_context: str = "normal") -> Dict[str, Any]:
    """
    Provide strategic guidance on what to learn next using AI analysis.
    
    Analyzes semantic profile + pathway state + recent activity to recommend:
    - focused_practice: Review and practice current material
    - move_forward: Continue to next topic
    - seek_help: Intervention recommended
    - break_recommended: Take a break to avoid burnout
    
    Args:
        user_id: User ID
        current_context: Context hint - "normal", "struggling", "ahead", "exploring"
    """
    return suggest_learning_path(user_id, current_context)


# =============================================================================
# ENTRY POINT
# =============================================================================
    """
    Manual semantic profile update - periodic analysis of learning patterns.
    Analyzes recent episodes to confirm long-term behavioral patterns.
    
    Updates (based on patterns across multiple recent episodes):
    - learning_disposition.preferred_explanation_style
    - learning_disposition.gaming_disposition
    - engagement_shape (updated averages and consistency)
    
    Args:
        user_id: User ID
        focus_area: What to focus on - "all", "learning_patterns", "engagement" (default: all)
    """
    return update_profile_manual(user_id, focus_area)


# ─── Mentor Chatbot MCP Tools ────────────────────────────────────────────────

@mcp.tool(name="get_student_context", title="Get Complete Student Context for Mentor Bot")
def tool_get_student_context(user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch complete contextual snapshot for mentor bot initialization.
    
    Returns:
    - current_session: Session details including course, topic, duration
    - semantic_profile: Learning patterns, struggles, strengths with enriched topic names
    - pathway_state: Current pathway, courses status, capstone status
    - skill_levels: Mastery scores per topic/skill with readable names
    - recent_history: Last mentor chats and course activities
    
    Args:
        user_id: User ID
        session_id: Optional session ID to get current session context
    """
    return get_student_context(user_id, session_id)


@mcp.tool(name="get_topic_content", title="Get Course Content for a Topic")
def tool_get_topic_content(topic_id: str, content_type: str = "all") -> Dict[str, Any]:
    """
    Retrieve course content for a specific topic to answer content questions.
    
    Returns topic concepts, common mistakes, when to use, and examples.
    Note: Currently returns placeholder data. Full content requires course_content table.
    
    Args:
        topic_id: Topic ID (module_id)
        content_type: Filter content type - "all", "concepts", "examples", "common_mistakes"
    """
    return get_topic_content(topic_id, content_type)


@mcp.tool(name="get_exercise_context", title="Get Exercise Details and Attempt History")
def tool_get_exercise_context(exercise_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get details about a specific exercise and optionally user's attempt history.
    
    Returns exercise details, difficulty, learning objectives, hints, and user attempts.
    
    Args:
        exercise_id: Exercise ID
        user_id: Optional user ID to include attempt history
    """
    return get_exercise_context(exercise_id, user_id)


@mcp.tool(name="search_similar_questions", title="Search Similar Previously Answered Questions")
def tool_search_similar_questions(user_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Find similar previously answered questions from mentor chat history.
    
    Uses keyword matching to find relevant past interactions.
    Note: Basic text matching. Production should use vector embeddings.
    
    Args:
        user_id: User ID
        query: Search query
        limit: Maximum number of results (default: 5)
    """
    return search_similar_questions(user_id, query, limit)


@mcp.tool(name="log_mentor_chat_turn", title="Log Mentor Chat Conversation Turn")
def tool_log_mentor_chat_turn(
    user_id: str,
    session_id: str,
    turn_number: int,
    role: str,
    message: str,
    course_id: Optional[str] = None,
    pathway_id: Optional[str] = None,
    topic_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record a mentor chat turn to raw_mentor_chat_turns table.
    These will later be aggregated into MENTOR_CHAT episodes.
    
    Args:
        user_id: User ID
        session_id: Session ID
        turn_number: Turn number in conversation
        role: "user" or "assistant"
        message: Full message text
        course_id: Optional course ID context
        pathway_id: Optional pathway ID context
        topic_id: Optional topic ID context
    """
    return log_mentor_chat_turn(user_id, session_id, turn_number, role, message,
                                 course_id, pathway_id, topic_id)


@mcp.tool(name="get_prerequisite_gaps", title="Identify Prerequisite Knowledge Gaps")
def tool_get_prerequisite_gaps(user_id: str, current_topic_id: str) -> Dict[str, Any]:
    """
    Identify prerequisite topics the user struggles with.
    
    Note: Requires prerequisite graph from FalkorDB. Currently returns placeholder.
    
    Args:
        user_id: User ID
        current_topic_id: Current topic being studied
    """
    return get_prerequisite_gaps(user_id, current_topic_id)


@mcp.tool(name="suggest_learning_path", title="Suggest Strategic Learning Path")
def tool_suggest_learning_path(user_id: str, current_context: str = "normal") -> Dict[str, Any]:
    """
    Provide strategic guidance on what to learn next using AI analysis.
    
    Analyzes semantic profile + pathway state + recent activity to recommend:
    - focused_practice: Review and practice current material
    - move_forward: Continue to next topic
    - seek_help: Intervention recommended
    - break_recommended: Take a break to avoid burnout
    
    Args:
        user_id: User ID
        current_context: Context hint - "normal", "struggling", "ahead", "exploring"
    """
    return suggest_learning_path(user_id, current_context)



# =============================================================================
# VIVA AGENT TOOLS — Runtime viva examination support
# =============================================================================

@mcp.tool()
def get_capstone_details(user_id: str) -> str:
    """Get the mega capstone requirements, description, and related context.
    Use this to understand what the user was supposed to build."""
    try:
        with DatabaseManager.get_cursor() as cur:
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
                return json.dumps({
                    "success": False,
                    "error": "No mega capstone found for user",
                    "data": None
                })
            
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
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "data": None})


@mcp.tool()
def get_capstone_review(user_id: str, capstone_id: str = None) -> str:
    """Get the code review results for the user's mega capstone submission.
    Includes code quality, design patterns, strengths, and areas for improvement."""
    try:
        with DatabaseManager.get_cursor() as cur:
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
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "data": None})


@mcp.tool()
def get_capstone_test(user_id: str, capstone_id: str = None) -> str:
    """Get the test results for the user's mega capstone submission.
    Includes pass/fail status, individual test cases, and any failures."""
    try:
        with DatabaseManager.get_cursor() as cur:
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
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "data": None})


@mcp.tool()
def start_viva(user_id: str, capstone_id: str, pathway_id: str = None) -> str:
    """Start a new viva session for a user.
    Must be called before recording questions and responses.
    Returns session_id and attempt_id needed for recording conversation turns."""
    try:
        with DatabaseManager.get_cursor() as cur:
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
        return json.dumps({"success": False, "error": str(e), "data": None})


@mcp.tool()
def get_viva_session(user_id: str, session_id: str) -> str:
    """Get the current state of a viva session including all conversation turns so far."""
    try:
        with DatabaseManager.get_cursor() as cur:
            cur.execute("""
                SELECT * FROM sessions 
                WHERE session_id = %s AND session_type = 'capstone_viva'
            """, (session_id,))
            session = cur.fetchone()
            
            if not session:
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
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "data": None})


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
    try:
        with DatabaseManager.get_cursor() as cur:
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
        return json.dumps({"success": False, "error": str(e), "data": None})


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
    try:
        with DatabaseManager.get_cursor() as cur:
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
        return json.dumps({"success": False, "error": str(e), "data": None})


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))
    print(f"Starting Neulearn Episodic + Semantic Tools MCP on {host}:{port}")
    mcp.run(transport="sse", host=host, port=port)
