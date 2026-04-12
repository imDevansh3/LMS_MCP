"""
Neulearn MCP Server - Mentor Tools + Viva Recording
===================================================

This MCP server provides:
1. 7 Mentor Chatbot Tools - for helping students during course study
2. 1 Viva Recording Tool - for logging viva conversation turns

All episodic and semantic processing has been moved to the neulearn-memory-service.
The Viva Agent uses this MCP server ONLY for recording turns - all examination
logic is handled by the agent itself with Kivor system prompts.

Transport: SSE (Server-Sent Events)
Port: 8001 (configurable via MCP_PORT env var)

Author: Neulearn Team
Date: 2026-04-09
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
from openai import AzureOpenAI
from fastmcp import FastMCP
from falkordb import FalkorDB as FalkorDBClient

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

FALKORDB_HOST  = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT  = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_GRAPH = os.getenv("FALKORDB_GRAPH", "neulearn")

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


DatabaseManager.initialize(DB_URL)


# =============================================================================
# FALKORDB CLIENT
# =============================================================================

class FalkorManager:
    _graph = None

    @classmethod
    def initialize(cls):
        if cls._graph is None:
            client = FalkorDBClient(host=FALKORDB_HOST, port=FALKORDB_PORT)
            cls._graph = client.select_graph(FALKORDB_GRAPH)
            logger.info(f"FalkorDB connected: graph={FALKORDB_GRAPH} at {FALKORDB_HOST}:{FALKORDB_PORT}")

    @classmethod
    def query(cls, cypher: str, params: dict = None) -> list:
        if cls._graph is None:
            raise RuntimeError("FalkorDB not initialized")
        result = cls._graph.query(cypher, params or {})
        return result.result_set  # list of rows


FalkorManager.initialize()


# =============================================================================
# OPENAI CLIENT
# =============================================================================

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)


# =============================================================================
# FASTMCP SERVER
# =============================================================================

mcp = FastMCP("Neulearn Mentor + Viva Recording Tools")


# =============================================================================
# MENTOR CHATBOT TOOLS (7 TOOLS)
# =============================================================================

@mcp.tool()
def get_student_context(user_id: str, session_id: Optional[str] = None) -> str:
    """
    Get complete student context for mentor bot including current progress,
    recent struggles, skill mastery, and learning patterns.
    
    Returns: JSON with student profile, current course, progress, skill gaps, and learning patterns
    """
    logger.info(f"get_student_context called: user_id={user_id}, session_id={session_id}")
    try:
        with DatabaseManager.get_cursor() as cur:
            # Get user basic info
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
            if not user:
                return json.dumps({
                    "success": False,
                    "error": f"User {user_id} not found",
                    "data": None
                })
            
            # Get current session if provided
            current_session = None
            current_course_id = None
            if session_id:
                cur.execute("""
                    SELECT * FROM sessions 
                    WHERE session_id = %s AND user_id = %s
                """, (session_id, user_id))
                current_session = cur.fetchone()
                if current_session:
                    current_course_id = current_session.get("course_id")
            
            # Get active pathway
            cur.execute("""
                SELECT * FROM student_pathways 
                WHERE user_id = %s AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))
            pathway = cur.fetchone()
            
            # Get current course progress
            course_progress = None
            if current_course_id:
                cur.execute("""
                    SELECT * FROM progress 
                    WHERE user_id = %s AND course_id = %s
                """, (user_id, current_course_id))
                course_progress = cur.fetchone()
            
            # Get skill mastery from Postgres
            cur.execute("""
                SELECT subtrack_id, mastery_score, confidence, last_updated
                FROM user_skill_mastery 
                WHERE user_id = %s
                ORDER BY mastery_score DESC
            """, (user_id,))
            mastery_rows = cur.fetchall()
            
            # Get recent semantic profile
            cur.execute("""
                SELECT * FROM semantic_profile_versions
                WHERE user_id = %s AND is_current = true
                LIMIT 1
            """, (user_id,))
            profile = cur.fetchone()
        
        # FalkorDB: resolve subtrack names and course name
        subtrack_names = {}
        course_name = None
        
        if mastery_rows:
            subtrack_ids = [r["subtrack_id"] for r in mastery_rows]
            falkor_result = FalkorManager.query("""
                MATCH (st:SubTrack)
                WHERE st.subtrack_id IN $ids
                RETURN st.subtrack_id, st.name
            """, {"ids": subtrack_ids})
            subtrack_names = {row[0]: row[1] for row in falkor_result}
        
        if current_course_id:
            falkor_result = FalkorManager.query("""
                MATCH (c:Course {course_id: $course_id})
                RETURN c.name
            """, {"course_id": current_course_id})
            course_name = falkor_result[0][0] if falkor_result else current_course_id
        
        # Extract meaningful insights from semantic profile
        learning_insights = None
        if profile and profile.get("profile"):
            prof_data = profile["profile"]
            
            # Extract identity insights
            identity = prof_data.get("identity", {})
            entry_assessment = identity.get("entry_assessment", {})
            
            # Extract skill insights from skill_map
            skill_map = prof_data.get("skill_map", {})
            subtracks = skill_map.get("subtracks", {})
            
            # Find strengths and struggles from modules
            strengths = []
            struggles = []
            for subtrack_name, subtrack_data in subtracks.items():
                modules = subtrack_data.get("modules", {})
                for mod_id, mod_data in modules.items():
                    if mod_data.get("mastery_score", 0) >= 0.7:
                        strengths.append({
                            "subtrack": subtrack_name,
                            "module": mod_id,
                            "mastery": mod_data.get("mastery_score"),
                            "level": mod_data.get("assessed_level")
                        })
                    elif mod_data.get("mastery_score", 0) < 0.5:
                        struggles.append({
                            "subtrack": subtrack_name,
                            "module": mod_id,
                            "mastery": mod_data.get("mastery_score"),
                            "struggle_areas": mod_data.get("areas_of_struggle", [])
                        })
            
            # Extract learning disposition
            learning_disp = prof_data.get("learning_disposition", {})
            calibration = learning_disp.get("calibration_profile", {})
            gaming = learning_disp.get("gaming_disposition", {})
            persistence = learning_disp.get("persistence_profile", {})
            
            # Extract engagement shape
            engagement = prof_data.get("engagement_shape", {})
            
            learning_insights = {
                "entry_level": entry_assessment.get("assessed_entry_level"),
                "calibration": {
                    "status": entry_assessment.get("calibration_at_entry"),
                    "trend": calibration.get("calibration_trend"),
                    "note": calibration.get("note")
                },
                "skill_strengths": strengths[:5],  # Top 5 strengths
                "skill_struggles": struggles[:5],  # Top 5 struggles
                "learning_patterns": {
                    "struggle_pattern": learning_disp.get("struggle_pattern"),
                    "learning_velocity": learning_disp.get("learning_velocity"),
                    "persistence": {
                        "gives_up_pattern": persistence.get("gives_up_pattern"),
                        "first_attempt_pass_rate": persistence.get("first_attempt_pass_rate"),
                        "note": persistence.get("note")
                    },
                    "gaming_risk": gaming.get("risk_level"),
                    "preferred_explanation_style": learning_disp.get("preferred_explanation_style")
                },
                "engagement": {
                    "consistency": engagement.get("consistency_label"),
                    "dropout_risk": engagement.get("dropout_risk_profile"),
                    "sessions_per_week": engagement.get("sessions_per_week_avg"),
                    "avg_session_duration_min": engagement.get("avg_session_duration_minutes")
                },
                "capstone_completed": len(prof_data.get("capstone_record", {}).get("completed", []))
            }
        
        # Build context
        context = {
            "user_id": user_id,
            "display_name": user["display_name"],
            "email": user["email"],
            "role": user["role"],
            "current_session": {
                "session_id": session_id,
                "course_id": current_course_id,
                "course_name": course_name,
                "started_at": current_session["started_at"].isoformat() if current_session and current_session.get("started_at") else None,
            } if current_session else None,
            "active_pathway": {
                "pathway_id": pathway["pathway_id"],
                "track_id": pathway["track_id"],
                "status": pathway["status"],
            } if pathway else None,
            "course_progress": {
                "course_id": course_progress["course_id"],
                "completion_percent": course_progress["completion_percent"],
                "total_sessions": course_progress["total_sessions"],
                "total_minutes": course_progress["total_minutes"],
            } if course_progress else None,
            "skill_mastery": [
                {
                    "subtrack_id": s["subtrack_id"],
                    "subtrack_name": subtrack_names.get(s["subtrack_id"], s["subtrack_id"]),
                    "mastery_score": float(s["mastery_score"]),
                    "confidence": s["confidence"],
                    "last_updated": s["last_updated"].isoformat() if s.get("last_updated") else None,
                }
                for s in mastery_rows
            ],
            "learning_insights": learning_insights,
        }
        
        return json.dumps({
            "success": True,
            "error": None,
            "data": context
        }, default=str)
            
    except Exception as e:
        logger.error(f"Error in get_student_context: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": None
        })


@mcp.tool()
def get_topic_content(course_id: str, user_id: str, content_type: str = "all") -> str:
    """
    Get learning content for a course including labels (theory/implementation content).
    Parameter renamed from topic_id to course_id to match actual schema.
    
    course_id: Course identifier
    user_id: Student identifier
    content_type: 'all' | 'theory' | 'exercises' - filter content type
    
    Returns: JSON with course name, labels, and exercises
    """
    logger.info(f"get_topic_content called: course_id={course_id}, user_id={user_id}, content_type={content_type}")
    try:
        # FalkorDB: First check if course exists and get its name
        course_result = FalkorManager.query("""
            MATCH (c:Course {course_id: $course_id})
            RETURN c.name, c.description
        """, {"course_id": course_id})

        if not course_result:
            return json.dumps({
                "success": False,
                "error": f"Course {course_id} not found in FalkorDB",
                "data": None
            })

        course_name = course_result[0][0]
        course_description = course_result[0][1] if len(course_result[0]) > 1 else None
        
        # FalkorDB: Now get all labels for this course (if any)
        labels_result = FalkorManager.query("""
            MATCH (c:Course {course_id: $course_id})-[:HAS_LABEL]->(l:Label)
            RETURN l.uid, l.name, l.label_type, 
                   l.ide_exercise_possible, l.exercise_id
            ORDER BY l.label_id
        """, {"course_id": course_id})

        labels = []
        exercise_ids = []

        for row in labels_result:
            uid, name, label_type, ide_possible, exercise_id = row
            label = {
                "uid": uid,
                "name": name,
                "label_type": label_type,  # 'theory' | 'implementation'
                "has_exercise": ide_possible == "yes",
                "exercise_id": exercise_id,
            }
            labels.append(label)
            if exercise_id:
                exercise_ids.append(exercise_id)

        # Filter by content_type if requested
        if content_type == "theory":
            labels = [l for l in labels if l["label_type"] == "theory"]
        elif content_type == "exercises":
            labels = [l for l in labels if l["has_exercise"]]

        # Postgres: get exercise details for labels that have exercises
        exercises = []
        if exercise_ids and content_type in ("all", "exercises"):
            with DatabaseManager.get_cursor() as cur:
                cur.execute("""
                    SELECT exercise_id, title, difficulty, pass_threshold,
                           problem_statement, hints
                    FROM exercises
                    WHERE exercise_id = ANY(%s)
                """, (exercise_ids,))
                exercises = [dict(e) for e in cur.fetchall()]

        return json.dumps({
            "success": True,
            "error": None,
            "data": {
                "course_id": course_id,
                "course_name": course_name,
                "course_description": course_description,
                "labels": labels,
                "exercises": exercises,
            }
        }, default=str)
            
    except Exception as e:
        logger.error(f"Error in get_topic_content: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": None
        })


@mcp.tool()
def get_exercise_context(exercise_id: str, user_id: str) -> str:
    """
    Get exercise details including problem statement, difficulty, hints, and user's attempt history.
    
    Returns: JSON with exercise details, attempt history, and course/label context
    """
    logger.info(f"get_exercise_context called: exercise_id={exercise_id}, user_id={user_id}")
    try:
        with DatabaseManager.get_cursor() as cur:
            # Get exercise details from Postgres
            cur.execute("""
                SELECT exercise_id, title, topic_id, module_id, course_id,
                       subtrack_id, difficulty, pass_threshold,
                       problem_statement, hints, test_cases
                FROM exercises 
                WHERE exercise_id = %s
            """, (exercise_id,))
            exercise = cur.fetchone()
            
            if not exercise:
                return json.dumps({
                    "success": False,
                    "error": f"Exercise {exercise_id} not found",
                    "data": None
                })
            
            # Get topic progress
            cur.execute("""
                SELECT * FROM topic_progress 
                WHERE user_id = %s AND topic_id = %s
            """, (user_id, exercise["topic_id"]))
            progress = cur.fetchone()
            
            # Get attempt history from raw_course_session
            cur.execute("""
                SELECT session_id, attempt_number, score, event_type, created_at
                FROM raw_course_session
                WHERE user_id = %s AND exercise_id = %s
                  AND event_type IN ('exercise_attempt', 'exercise_pass', 'exercise_fail')
                ORDER BY created_at ASC
            """, (user_id, exercise_id))
            attempts = [dict(a) for a in cur.fetchall()]
        
        # FalkorDB: resolve course name and label name
        course_name = None
        label_name = None
        if exercise["course_id"]:
            falkor_result = FalkorManager.query("""
                MATCH (c:Course {course_id: $course_id})-[:HAS_LABEL]->(l:Label)
                WHERE l.exercise_id = $exercise_id
                RETURN c.name, l.name
            """, {"course_id": exercise["course_id"], "exercise_id": exercise_id})
            
            if falkor_result:
                course_name = falkor_result[0][0]
                label_name = falkor_result[0][1]
        
        return json.dumps({
            "success": True,
            "error": None,
            "data": {
                "exercise_id": exercise_id,
                "title": exercise["title"],
                "course_id": exercise["course_id"],
                "course_name": course_name,
                "label_name": label_name,
                "topic_id": exercise["topic_id"],
                "subtrack_id": exercise["subtrack_id"],
                "problem_statement": exercise["problem_statement"],
                "difficulty": exercise["difficulty"],
                "pass_threshold": exercise["pass_threshold"],
                "hints": exercise["hints"],
                "test_cases": exercise["test_cases"],
                "progress": {
                    "best_score": progress["best_score"],
                    "passed": progress["passed"],
                    "status": progress["status"],
                } if progress else None,
                "attempts": attempts,
            }
        }, default=str)
            
    except Exception as e:
        logger.error(f"Error in get_exercise_context: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": None
        })


@mcp.tool()
def search_similar_questions(query: str, user_id: str, course_id: str = None, limit: int = 5) -> str:
    """
    Search for similar questions from past mentor chat sessions to provide context.
    Uses simple text similarity on past mentor chat episodes.
    
    Returns: JSON with similar questions and their resolutions
    """
    logger.info(f"search_similar_questions called: user_id={user_id}, query={query[:50]}..., course_id={course_id}")
    try:
        with DatabaseManager.get_cursor() as cur:
            # Search mentor chat episodes - note: fixed episode type to kebab-case
            if course_id:
                cur.execute("""
                    SELECT * FROM episodic_episodes
                    WHERE user_id = %s AND type = 'mentor-chat'
                      AND data->>'course_id' = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (user_id, course_id, limit * 2))
            else:
                cur.execute("""
                    SELECT * FROM episodic_episodes
                    WHERE user_id = %s AND type = 'mentor-chat'
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (user_id, limit * 2))
            episodes = cur.fetchall()
            
            # Simple keyword matching (in production, use embeddings)
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            results = []
            for ep in episodes:
                data = ep["data"]
                if isinstance(data, str):
                    data = json.loads(data)
                
                # Check if any query words appear in the episode
                episode_text = json.dumps(data).lower()
                if any(word in episode_text for word in query_words):
                    results.append({
                        "episode_id": ep["episode_id"],
                        "timestamp": ep["timestamp"].isoformat() if ep.get("timestamp") else None,
                        "summary": data.get("summary", data.get("agent_summary", "")),
                        "unresolved_topics": data.get("unresolved_topics", []),
                        "topics_discussed": data.get("topics_discussed", []),
                    })
                
                if len(results) >= limit:
                    break
            
            return json.dumps({
                "success": True,
                "error": None,
                "data": {
                    "query": query,
                    "results": results,
                }
            }, default=str)
            
    except Exception as e:
        logger.error(f"Error in search_similar_questions: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": None
        })


@mcp.tool()
def log_mentor_chat_turn(
    user_id: str,
    session_id: str,
    course_id: str,
    pathway_id: Optional[str],
    turn_number: int,
    role: str,
    message: str
) -> str:
    """
    Log a single turn in a mentor chat conversation.
    Call this for every message exchanged between mentor bot and student.
    
    role: 'user' or 'assistant' (NOT 'viva_agent' - that's for viva only)
    turn_number: sequential number starting from 1
    
    Returns: JSON with turn_id and confirmation
    """
    logger.info(f"log_mentor_chat_turn called: user_id={user_id}, session_id={session_id}, turn={turn_number}")
    
    # Validate role
    if role not in ("user", "assistant"):
        return json.dumps({
            "success": False,
            "error": f"Invalid role '{role}'. Must be 'user' or 'assistant'",
            "data": None
        })
    
    try:
        with DatabaseManager.get_cursor() as cur:
            cur.execute("""
                INSERT INTO raw_mentor_chat_turns 
                (user_id, session_id, course_id, pathway_id, turn_number, role, message)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
            """, (user_id, session_id, course_id, pathway_id, turn_number, role, message))
            turn = cur.fetchone()
            
            return json.dumps({
                "success": True,
                "error": None,
                "data": {
                    "turn_id": turn["id"],
                    "turn_number": turn_number,
                    "created_at": turn["created_at"].isoformat() if turn.get("created_at") else None,
                }
            }, default=str)
            
    except Exception as e:
        logger.error(f"Error in log_mentor_chat_turn: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": None
        })


@mcp.tool()
def get_prerequisite_gaps(user_id: str, course_id: str) -> str:
    """
    Identify prerequisite skill gaps that might be blocking the student's understanding.
    Parameter renamed from topic_id to course_id to match actual schema.
    Checks user_skill_mastery for the subtrack this course belongs to.
    
    Returns: JSON with identified gaps and recommendations
    """
    logger.info(f"get_prerequisite_gaps called: user_id={user_id}, course_id={course_id}")
    try:
        # FalkorDB: get the subtrack this course belongs to
        falkor_result = FalkorManager.query("""
            MATCH (t:Track)-[:HAS_SUBTRACK]->(st:SubTrack)-[:HAS_COURSE]->(c:Course {course_id: $course_id})
            RETURN c.name, st.subtrack_id, st.name, t.track_id, t.name
        """, {"course_id": course_id})

        if not falkor_result:
            return json.dumps({
                "success": False,
                "error": f"Course {course_id} not found in FalkorDB",
                "data": None
            })

        course_name, subtrack_id, subtrack_name, track_id, track_name = falkor_result[0]

        # Postgres: get user's mastery for this subtrack
        with DatabaseManager.get_cursor() as cur:
            cur.execute("""
                SELECT mastery_score, confidence, last_updated
                FROM user_skill_mastery
                WHERE user_id = %s AND subtrack_id = %s
            """, (user_id, subtrack_id))
            mastery = cur.fetchone()

        # Identify if this is a gap (low mastery score)
        gaps = []
        if mastery and mastery["mastery_score"] < 0.5:
            gaps.append({
                "subtrack_id": subtrack_id,
                "subtrack_name": subtrack_name,
                "mastery_score": float(mastery["mastery_score"]),
                "severity": "high" if mastery["mastery_score"] < 0.3 else "medium",
                "recommendation": "Review foundational concepts before continuing",
            })

        return json.dumps({
            "success": True,
            "error": None,
            "data": {
                "course_id": course_id,
                "course_name": course_name,
                "subtrack_id": subtrack_id,
                "subtrack_name": subtrack_name,
                "track_id": track_id,
                "track_name": track_name,
                "current_mastery": float(mastery["mastery_score"]) if mastery else None,
                "confidence": mastery["confidence"] if mastery else None,
                "gaps": gaps,
                "note": "Full prerequisite traversal requires prerequisite edges in FalkorDB"
            }
        }, default=str)
            
    except Exception as e:
        logger.error(f"Error in get_prerequisite_gaps: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": None
        })


@mcp.tool()
def suggest_learning_path(user_id: str, current_course_id: str) -> str:
    """
    Suggest next steps based on the student's current position and skill gaps.
    Parameter renamed from current_topic_id to current_course_id to match schema.
    
    Returns: JSON with suggested next steps
    """
    logger.info(f"suggest_learning_path called: user_id={user_id}, current_course_id={current_course_id}")
    try:
        with DatabaseManager.get_cursor() as cur:
            # Get active pathway
            cur.execute("""
                SELECT pathway_id, track_id FROM student_pathways 
                WHERE user_id = %s AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))
            pathway = cur.fetchone()
            
            if not pathway:
                return json.dumps({
                    "success": False,
                    "error": "No active pathway found for user",
                    "data": None
                })
            
            # Get upcoming courses in this pathway
            cur.execute("""
                SELECT course_id, position FROM enrollments 
                WHERE user_id = %s AND pathway_id = %s
                  AND completed_at IS NULL
                ORDER BY position
                LIMIT 5
            """, (user_id, pathway["pathway_id"]))
            upcoming_enrollments = cur.fetchall()
            
            # Get recent course activity episodes for momentum
            cur.execute("""
                SELECT * FROM episodic_episodes
                WHERE user_id = %s AND type = 'course-activity'
                ORDER BY timestamp DESC
                LIMIT 5
            """, (user_id,))
            recent_activity = cur.fetchall()
        
        # FalkorDB: enrich with course names
        next_courses = []
        if upcoming_enrollments:
            course_ids = [e["course_id"] for e in upcoming_enrollments]
            falkor_result = FalkorManager.query("""
                MATCH (c:Course)
                WHERE c.course_id IN $ids
                RETURN c.course_id, c.name
            """, {"ids": course_ids})
            course_names = {row[0]: row[1] for row in falkor_result}
            
            for enroll in upcoming_enrollments:
                cid = enroll["course_id"]
                next_courses.append({
                    "course_id": cid,
                    "course_name": course_names.get(cid, cid),
                    "position": enroll["position"],
                })
        
        suggestions = {
            "current_course_id": current_course_id,
            "active_pathway_id": pathway["pathway_id"],
            "track_id": pathway["track_id"],
            "upcoming_courses": next_courses,
            "recent_activity_count": len(recent_activity),
            "recommendation": "Continue with next course in pathway" if next_courses else "Pathway complete",
        }
        
        return json.dumps({
            "success": True,
            "error": None,
            "data": suggestions
        }, default=str)
            
    except Exception as e:
        logger.error(f"Error in suggest_learning_path: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": None
        })


# =============================================================================
# VIVA RECORDING TOOL (SIMPLIFIED)
# =============================================================================

@mcp.tool()
def record_viva_turn(
    user_id: str,
    session_id: str,
    capstone_id: str,
    attempt_id: str,
    turn_number: int,
    role: str,
    message: str,
    pathway_id: Optional[str] = None
) -> str:
    """
    Record a single viva conversation turn to raw_viva_turns.
    Call this for every message exchanged — both viva_agent questions 
    and user responses. turn_number starts at 1 and increments by 1 
    for every message.
    
    role: 'viva_agent' for examiner questions, 'user' for student responses
          (NOT 'assistant' - that's for mentor chat only)
    turn_number: sequential turn number (1, 2, 3, ...)
    message: the full text of the message
    
    Returns: JSON with turn_id and confirmation
    """
    logger.info(f"record_viva_turn called: user_id={user_id}, session_id={session_id}, turn={turn_number}, role={role}")
    
    # Validate role
    if role not in ("viva_agent", "user"):
        return json.dumps({
            "success": False,
            "error": f"Invalid role '{role}'. Must be 'viva_agent' or 'user'",
            "data": None
        })
    
    try:
        with DatabaseManager.get_cursor() as cur:
            cur.execute("""
                INSERT INTO raw_viva_turns 
                (user_id, session_id, capstone_id, pathway_id, attempt_id, turn_number, role, message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
            """, (user_id, session_id, capstone_id, pathway_id, attempt_id, turn_number, role, message))
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
                    "created_at": turn["created_at"].isoformat() if turn.get("created_at") else None,
                }
            }, default=str)
            
    except Exception as e:
        logger.error(f"Error in record_viva_turn: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "data": None
        })


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))
    logger.info(f"Starting Neulearn Mentor + Viva Recording MCP Server on {host}:{port}")
    logger.info("Available tools: 7 mentor tools + 1 viva recording tool")
    mcp.run(transport="sse", host=host, port=port)
