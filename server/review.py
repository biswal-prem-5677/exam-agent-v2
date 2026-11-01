"""
review.py – Review queue management and audit logging helpers.
Provides business logic for human-in-the-loop review and dispute handling.
"""
from typing import Dict, List, Optional
from server.db import insert_review, get_reviews, get_held_sessions, log_agent_step


def process_review_action(
    session_id: str,
    action: str,
    note: str,
    override_grade: Optional[float] = None,
    reviewer: str = "examiner"
) -> Dict[str, any]:
    """
    Validate and execute a human review action:
    action: 'approve' | 'override' | 'mark_cheat' | 'dismiss' | 'appeal'
    """
    if not note or not note.strip():
        raise ValueError("A mandatory explanation note is required for every review action.")
    
    valid_actions = {"approve", "override", "mark_cheat", "dismiss", "appeal"}
    if action not in valid_actions:
        raise ValueError(f"Invalid review action: '{action}'. Must be one of {valid_actions}")

    if action == "override":
        if override_grade is None or not (0.0 <= override_grade <= 5.0):
            raise ValueError("Override grade must be a number between 0.0 and 5.0")

    insert_review(
        session_id=session_id,
        action=action,
        note=note.strip(),
        override_grade=override_grade,
        reviewer=reviewer
    )
    
    log_agent_step(
        session_id=session_id,
        step="human_review",
        detail=f"Action: {action}, Note: {note}" + (f", Grade: {override_grade}" if override_grade is not None else ""),
        model_ver="human"
    )

    return {
        "success": True,
        "session_id": session_id,
        "action": action,
        "reviewer": reviewer
    }


def list_pending_reviews() -> List[Dict]:
    """Return all held sessions waiting for human review."""
    return get_held_sessions()


def get_session_review_history(session_id: str) -> List[Dict]:
    """Return audit history of reviews for a given session."""
    return get_reviews(session_id)

