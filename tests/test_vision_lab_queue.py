"""The queue: a job that is waiting is never lost, and a worker that is not running
is never silent.

THE BUG THESE PIN
    "This job's worker stopped reporting" was applied to QUEUED jobs as well,
    whose only timestamp is their creation time. Any job that waited 30 minutes
    behind a backlog was failed as processing_interrupted without ever being
    analysed. At 60-100 s per ad, the back of a 20-30 ad queue was thrown away,
    and it looked like a routine failure rather than what it was.

    /health had the matching blind spot: it read worker liveness from every
    active job, queued ones included, so one freshly queued job made a worker
    that had been dead for hours look healthy.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

# Fixtures and the configured app come from the /analyze suite.
from test_vision_lab_api import (  # noqa: F401 - fixtures are used by name
    BODY,
    HEADERS,
    client,
    wired,
    work_the_queue,
)

import vision_lab as vl
from vision_lab import store as st

LONG_AGO = timedelta(hours=2)


def submit(client, creative_id):
    body = dict(BODY, creative_id=creative_id)
    return client.post("/api/vision-lab/analyze", json=body,
                       headers=HEADERS).json()["analysis_id"]


def age(store, analysis_id, by):
    """Pretend a job was created (and last heartbeated) `by` ago."""
    old = st.now_utc() - by
    asyncio.run(store.collection.update_one(
        {"_id": analysis_id}, {"$set": {"created_at": old, "heartbeat_at": old}}))


def get(store, analysis_id):
    return asyncio.run(store.get(analysis_id))


def health(client):
    return client.get("/health").json()["vision_lab"]


# =========================================================================== #
# The rule
# =========================================================================== #
def test_a_job_waiting_in_the_queue_is_never_stale():
    waiting = {"status": vl.STATUS_QUEUED,
               "created_at": st.now_utc() - LONG_AGO,
               "heartbeat_at": st.now_utc() - LONG_AGO}
    assert st.is_stale(waiting) is False


def test_a_started_job_whose_worker_went_quiet_still_is():
    """What staleness is FOR is unchanged: a worker that died mid-analysis."""
    for status in vl.CLAIMED_STATUSES:
        dead = {"status": status, "heartbeat_at": st.now_utc() - LONG_AGO}
        assert st.is_stale(dead) is True, status


def test_queued_is_the_only_active_status_no_worker_holds():
    assert set(vl.ACTIVE_STATUSES) - set(vl.CLAIMED_STATUSES) == {vl.STATUS_QUEUED}


# =========================================================================== #
# The two places that used to fail waiting jobs
# =========================================================================== #
def test_the_worker_reaper_leaves_a_long_queued_job_alone(wired, client):
    from vision_lab.worker import reap_stale

    started = submit(client, "cre_started")
    asyncio.run(wired.claim_next("worker-that-died"))     # takes `started`
    waiting = submit(client, "cre_waiting")
    age(wired, started, LONG_AGO)                         # its worker went quiet
    age(wired, waiting, LONG_AGO)                         # it has simply waited

    assert asyncio.run(reap_stale(wired)) == 1

    # The dead job is reported, as before...
    assert get(wired, started)["reason"] == vl.PROCESSING_INTERRUPTED
    # ...and the waiting one is untouched - still queued, still holding the link
    # the worker needs to download it.
    still = get(wired, waiting)
    assert still["status"] == vl.STATUS_QUEUED
    assert still.get("creative_url")


def test_polling_a_long_queued_job_does_not_fail_it(wired, client):
    """GET reaps stale jobs too - it must not reap a waiting one."""
    waiting = submit(client, "cre_waiting")
    age(wired, waiting, LONG_AGO)

    body = client.get(f"/api/vision-lab/analysis/{waiting}", headers=HEADERS).json()
    assert body["status"] == vl.STATUS_QUEUED
    assert body["availability"]["available"] is True


def test_a_long_queued_job_is_still_analysed_when_its_turn_comes(wired, client):
    """The whole point: it is not merely spared, it gets done."""
    waiting = submit(client, "cre_waiting")
    age(wired, waiting, LONG_AGO)

    work_the_queue(wired)
    assert get(wired, waiting)["status"] == vl.STATUS_COMPLETED


def test_resubmitting_a_long_queued_job_reuses_it(wired, client):
    """A waiting job is still a valid answer - resubmitting must not start a
    second, paid analysis of the same ad."""
    first = submit(client, "cre_same")
    age(wired, first, LONG_AGO)

    again = client.post("/api/vision-lab/analyze", json=dict(BODY, creative_id="cre_same"),
                        headers=HEADERS).json()
    assert again["analysis_id"] == first
    assert again["idempotent_hit"] is True


# =========================================================================== #
# /health - where a missing worker now shows
# =========================================================================== #
def test_health_flags_jobs_nobody_is_working_on(wired, client):
    """The failure that used to be silent: no worker running at all."""
    waiting = submit(client, "cre_waiting")
    age(wired, waiting, timedelta(minutes=5))

    h = health(client)
    assert h["queue"]["queued"] == 1
    assert h["queue"]["oldest_waiting_seconds"] >= 300
    assert h["worker"]["in_flight"] == 0
    assert h["worker"]["healthy"] is False
    assert h["worker"]["reason"] == "jobs_waiting_unclaimed"


def test_a_busy_worker_with_a_backlog_is_healthy(wired, client):
    """A long queue behind a working worker is load, not failure - and waiting
    longer than the old 30-minute cut-off must not change that."""
    submit(client, "cre_being_processed")
    asyncio.run(wired.claim_next("worker-a"))               # heartbeat: now
    for n in range(3):
        age(wired, submit(client, f"cre_backlog_{n}"), timedelta(minutes=40))

    h = health(client)
    assert h["worker"]["healthy"] is True
    assert h["worker"]["in_flight"] == 1                    # queued is not in flight
    assert h["queue"]["queued"] == 3
    assert "reason" not in h["worker"]


def test_a_fresh_queued_job_cannot_make_a_dead_worker_look_alive(wired, client):
    """REGRESSION. Liveness was read from every active job's heartbeat, and a
    queued job's heartbeat is its creation time - so one job queued a second ago
    reported a worker that died two hours ago as healthy."""
    started = submit(client, "cre_started")
    asyncio.run(wired.claim_next("worker-that-died"))
    age(wired, started, LONG_AGO)
    submit(client, "cre_queued_just_now")

    assert health(client)["worker"]["healthy"] is False


def test_an_empty_queue_is_unknown_not_unhealthy(wired, client):
    h = health(client)
    assert h["worker"]["healthy"] is None
    assert h["queue"] == {"queued": 0, "oldest_waiting_seconds": None}
