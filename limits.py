"""DB-backed message limit enforcement.

Plans:
  guest  -> 5 messages / day  (no account)
  free   -> 20 messages / day (logged in, no subscription)
  pro    -> unlimited          (active Paddle subscription)
"""

import datetime

PLAN_LIMITS: dict[str, int | float] = {
    "guest": 5,
    "free":  20,
    "pro":   float("inf"),
}


def get_user_plan(user_id: str, sb) -> str:
    """Return the plan string for a logged-in Supabase user."""
    if not sb or not user_id:
        return "guest"
    try:
        res = sb.table("profiles").select("plan").eq("id", user_id).single().execute()
        return (res.data or {}).get("plan", "free")
    except Exception:
        return "free"


def check_and_increment(
    session_id: str,
    user_id: str | None,
    sb,
    plan: str | None = None,
) -> tuple[bool, int, int | None]:
    """Check quota and increment counter atomically.

    Returns:
        (allowed, messages_used_today, daily_limit_or_None_if_unlimited)
    """
    today = datetime.date.today().isoformat()

    if plan is None:
        plan = get_user_plan(user_id, sb) if user_id else "guest"

    raw_limit = PLAN_LIMITS.get(plan, 20)
    limit_int  = None if raw_limit == float("inf") else int(raw_limit)

    # Track by user_id when logged in, session_id for guests
    tracking_id = user_id if user_id else session_id

    if not sb:
        return True, 0, limit_int  # no DB — allow all

    try:
        res = (
            sb.table("usage")
            .select("count")
            .eq("user_id", tracking_id)
            .eq("date", today)
            .execute()
        )
        if res.data:
            count = res.data[0]["count"]
            if limit_int is not None and count >= limit_int:
                return False, count, limit_int
            sb.table("usage").update({"count": count + 1}).eq("user_id", tracking_id).eq("date", today).execute()
            return True, count + 1, limit_int
        else:
            if limit_int is not None and limit_int <= 0:
                return False, 0, limit_int
            sb.table("usage").insert({"user_id": tracking_id, "date": today, "count": 1}).execute()
            return True, 1, limit_int
    except Exception as e:
        print(f"[limits] DB error: {e}")
        return True, 0, limit_int  # fail open


def get_usage_today(session_id: str, user_id: str | None, sb) -> dict:
    """Return a summary dict for the usage/session endpoint."""
    today  = datetime.date.today().isoformat()
    plan   = get_user_plan(user_id, sb) if user_id else "guest"
    raw    = PLAN_LIMITS.get(plan, 20)
    limit_int = None if raw == float("inf") else int(raw)
    tracking_id = user_id if user_id else session_id

    count = 0
    try:
        if sb:
            res = sb.table("usage").select("count").eq("user_id", tracking_id).eq("date", today).execute()
            count = res.data[0]["count"] if res.data else 0
    except Exception:
        pass

    remaining = None if limit_int is None else max(0, limit_int - count)
    return {
        "plan":      plan,
        "used":      count,
        "limit":     limit_int,
        "remaining": remaining,
    }
