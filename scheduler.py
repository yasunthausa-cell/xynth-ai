"""Persistent task scheduler for Xynth AI.

A single APScheduler instance runs in-process. Jobs are persisted to SQLite so
they survive workflow restarts. When a job fires, it runs a prompt through the
agent and sends the answer to the user via WhatsApp.
"""
import os
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

_scheduler = None
_run_agent_fn = None
_send_whatsapp_fn = None


def _execute_scheduled_job(job_id: str, prompt: str, recipient: str):
    """Top-level so APScheduler can pickle/load it from the SQLite jobstore."""
    if _run_agent_fn is None or _send_whatsapp_fn is None:
        print(f"[scheduler] Job {job_id} fired but app not initialised yet.")
        return
    session_id = f"scheduled-{job_id}-{int(datetime.datetime.utcnow().timestamp())}"
    print(f"[scheduler] Running job {job_id} for {recipient}")
    try:
        result = _run_agent_fn(session_id, prompt)
    except Exception as e:
        result = f"Scheduled task failed: {e}"
    _send_whatsapp_fn(recipient, f"⏰ Scheduled task '{job_id}':\n\n{result}")


def init_scheduler(run_agent_fn, send_whatsapp_fn):
    """Wire the scheduler to the agent + WhatsApp sender. Call once at startup."""
    global _scheduler, _run_agent_fn, _send_whatsapp_fn
    _run_agent_fn = run_agent_fn
    _send_whatsapp_fn = send_whatsapp_fn
    if _scheduler is not None:
        return _scheduler
    jobstores = {"default": SQLAlchemyJobStore(url="sqlite:///scheduled_jobs.db")}
    _scheduler = BackgroundScheduler(jobstores=jobstores, timezone="UTC")
    _scheduler.start()
    print(f"[scheduler] Started with {len(_scheduler.get_jobs())} existing jobs.")
    return _scheduler


def schedule_task(prompt: str, recipient: str, hour: int, minute: int = 0,
                  day_of_week: str = "*", timezone: str = "UTC") -> str:
    """Schedule a recurring task. Returns a confirmation string with the job_id."""
    if _scheduler is None:
        return "Scheduler not initialised."
    job_id = f"job_{int(datetime.datetime.utcnow().timestamp() * 1000)}"
    try:
        trigger = CronTrigger(hour=hour, minute=minute, day_of_week=day_of_week, timezone=timezone)
    except Exception as e:
        return f"Invalid schedule: {e}"
    _scheduler.add_job(
        _execute_scheduled_job,
        trigger=trigger,
        args=[job_id, prompt, recipient],
        id=job_id,
        replace_existing=True,
        misfire_grace_time=300,
    )
    job = _scheduler.get_job(job_id)
    return (f"✅ Scheduled '{job_id}' to run at {hour:02d}:{minute:02d} ({timezone}) "
            f"on day(s) '{day_of_week}'. Next run: {job.next_run_time}.")


def schedule_one_time_task(prompt: str, recipient: str, run_at_iso: str,
                           timezone: str = "UTC") -> str:
    """Schedule a single-fire task at the given ISO datetime (e.g. '2026-04-29T14:30')."""
    if _scheduler is None:
        return "Scheduler not initialised."
    try:
        import pytz
        dt = datetime.datetime.fromisoformat(run_at_iso)
        tz = pytz.timezone(timezone)
        if dt.tzinfo is None:
            dt = tz.localize(dt)
    except Exception as e:
        return f"Invalid date/timezone: {e}"
    job_id = f"job_{int(datetime.datetime.utcnow().timestamp() * 1000)}"
    _scheduler.add_job(
        _execute_scheduled_job,
        trigger=DateTrigger(run_date=dt),
        args=[job_id, prompt, recipient],
        id=job_id,
        misfire_grace_time=300,
    )
    return f"✅ Scheduled one-time job '{job_id}' to run at {dt.isoformat()}."


def list_tasks(recipient_filter: str = None) -> str:
    if _scheduler is None:
        return "Scheduler not initialised."
    jobs = _scheduler.get_jobs()
    if recipient_filter:
        jobs = [j for j in jobs if len(j.args) >= 3 and j.args[2] == recipient_filter]
    if not jobs:
        return "No scheduled tasks."
    lines = []
    for j in jobs:
        prompt = j.args[1] if len(j.args) > 1 else "?"
        recip = j.args[2] if len(j.args) > 2 else "?"
        lines.append(f"• {j.id} | next: {j.next_run_time} | to: {recip} | task: {prompt[:80]}")
    return "\n".join(lines)


def cancel_task(job_id: str) -> str:
    if _scheduler is None:
        return "Scheduler not initialised."
    try:
        _scheduler.remove_job(job_id)
        return f"✅ Cancelled task '{job_id}'."
    except Exception as e:
        return f"Failed to cancel '{job_id}': {e}"
