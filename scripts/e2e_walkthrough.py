"""End-to-end walk of the whole product against a running server.

Drives the real HTTP API the way a client would, from registering a tenant to
downloading both PDFs, and prints PASS/FAIL per step so a failure is
attributable to one call rather than to "the pipeline".

Needs the API, a Celery worker, and the backing services up -- see
docs/manual-testing.md. It calls the real NVIDIA endpoints, so a full run costs
a plan generation plus one model call per rubric criterion and takes a few
minutes.

    python scripts/e2e_walkthrough.py

Two steps reach past HTTP and into the process: the interview lifecycle is
driven by publishing SessionStarted/SessionEnded on the event bus, because the
alternative is a real WebRTC call with a real microphone. Everything else is
ordinary HTTP.
"""

import asyncio
import io
import os
import sys
import time
import uuid

import httpx

BASE = os.environ.get("AII_BASE_URL", "http://localhost:8000") + "/api/v1"
SLUG = f"e2e{uuid.uuid4().hex[:8]}"
PW = "correct-horse-battery-staple"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def _pdf(text: str) -> bytes:
    """A real one-page PDF, so the parser has something to parse."""
    from pypdf import PdfWriter
    from reportlab.pdfgen import canvas as rl_canvas

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf)
    y = 800
    for line in text.split("\n"):
        c.drawString(60, y, line[:95])
        y -= 14
    c.save()
    buf.seek(0)
    writer = PdfWriter(clone_from=buf)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


RESUME = """Ada Lovelace
Senior Backend Engineer

EXPERIENCE
Staff Engineer, Northwind Data (2021-2026)
Led the migration of a single-tenant Postgres platform to a sharded
multi-tenant design. Sharded on tenant id; moved cross-tenant reporting
to a nightly ClickHouse rollup. Two engineers, three months, dual-write
with a kill switch on the read path.
Built an async Python ingestion pipeline: FastAPI, Celery, asyncpg,
processing 40M events per day.

Backend Engineer, Contoso (2018-2021)
Owned the billing service. Introduced idempotency keys after a duplicate
charge incident; wrote the reconciliation job that found the other 14.

SKILLS
Python, FastAPI, Postgres, SQLAlchemy, Celery, Redis, Kafka, ClickHouse,
Docker, Kubernetes, distributed systems, database internals.
"""

JD = """We are hiring a senior backend engineer for a multi-tenant Python
platform: FastAPI, Postgres, async SQLAlchemy, Celery and S3-compatible
object storage.

You will own service boundaries end to end, from schema design through to
the queries that run in production. We care most about how you reason about
tradeoffs -- consistency against latency, isolation against complexity --
and whether you can explain a decision you made and what it cost you.

Requirements: strong Python, real experience with relational data modelling
and concurrency, and comfort operating what you build."""


async def main() -> None:
    async with httpx.AsyncClient(timeout=60.0) as c:
        print("\n=== 1. Ops ===")
        r = await c.get(BASE.removesuffix("/api/v1") + "/health")
        check("GET /health", r.status_code == 200)
        r = await c.get(BASE.removesuffix("/api/v1") + "/ready")
        check("GET /ready", r.status_code == 200, str(r.json()["checks"]))
        r = await c.get(BASE.removesuffix("/api/v1") + "/metrics")
        check("GET /metrics", r.status_code == 200, str(r.json()))

        print("\n=== 2. Register + login ===")
        r = await c.post(
            f"{BASE}/auth/register-org",
            json={
                "org_name": "E2E Co",
                "slug": SLUG,
                "admin_email": f"admin@{SLUG}.example.com",
                "admin_password": PW,
            },
        )
        if not check("POST /auth/register-org", r.status_code == 201, r.text[:200]):
            return
        org = r.json()
        H = {"Authorization": f"Bearer {org['tokens']['access_token']}"}

        r = await c.post(
            f"{BASE}/auth/login",
            json={"email": f"admin@{SLUG}.example.com", "password": PW},
        )
        check("POST /auth/login", r.status_code == 200)

        r = await c.post(
            f"{BASE}/auth/refresh", json={"refresh_token": r.json()["refresh_token"]}
        )
        check("POST /auth/refresh (rotation)", r.status_code == 200)

        r = await c.get(f"{BASE}/auth/me", headers=H)
        check("GET /auth/me", r.status_code == 200, r.json().get("role", ""))

        print("\n=== 3. Job + description ===")
        r = await c.post(f"{BASE}/jobs", headers=H, json={"title": "Senior Backend Engineer"})
        if not check("POST /jobs", r.status_code == 201, r.text[:200]):
            return
        job_id = r.json()["id"]

        r = await c.post(
            f"{BASE}/jobs/{job_id}/descriptions", headers=H, json={"content": JD}
        )
        check("POST /jobs/{id}/descriptions", r.status_code == 201, r.text[:200])

        print("\n=== 4. Proctoring policy ===")
        r = await c.put(
            f"{BASE}/jobs/{job_id}/proctoring-policy",
            headers=H,
            json={"blur_limit": 2, "fullscreen_required": False, "auto_terminate": False},
        )
        check("PUT proctoring-policy", r.status_code == 200, r.text[:200])

        print("\n=== 5. Invite + redeem ===")
        cand_email = f"ada@{SLUG}.example.com"
        r = await c.post(
            f"{BASE}/auth/invites",
            headers=H,
            json={"candidate_email": cand_email, "candidate_name": "Ada Lovelace",
                  "job_id": job_id},
        )
        if not check("POST /auth/invites", r.status_code == 201, r.text[:200]):
            return
        inv = r.json()
        interview_id = inv["interview_id"]

        r = await c.post(
            f"{BASE}/auth/invites/redeem", json={"invite_token": inv["invite_token"]}
        )
        if not check("POST /auth/invites/redeem", r.status_code == 200, r.text[:200]):
            return
        CH = {"Authorization": f"Bearer {r.json()['interview_token']}"}

        r = await c.get(f"{BASE}/interviews/me", headers=CH)
        check("GET /interviews/me (candidate)", r.status_code == 200, r.json().get("status", ""))

        print("\n=== 6. Resume upload (candidate) ===")
        r = await c.post(
            f"{BASE}/candidates/me/resume/presign",
            headers=CH,
            json={"filename": "ada.pdf", "content_type": "application/pdf"},
        )
        if not check("POST resume/presign", r.status_code in (200, 201), r.text[:120]):
            return
        pre = r.json()

        pdf = _pdf(RESUME)
        put = await c.put(
            pre["upload_url"], content=pdf, headers={"Content-Type": "application/pdf"}
        )
        check("PUT to S3", put.status_code in (200, 204), f"{put.status_code}")

        r = await c.post(
            f"{BASE}/candidates/me/resume/{pre['resume_id']}/complete", headers=CH
        )
        check("POST resume/complete", r.status_code == 200, r.json().get("status", ""))

        print("\n  waiting for resume parse + embeddings...")
        parsed = False
        for _ in range(40):
            await asyncio.sleep(3)
            r = await c.get(f"{BASE}/candidates/me/resume", headers=CH)
            body = r.json() if r.status_code == 200 else []
            items = body.get("items", []) if isinstance(body, dict) else body
            status = items[0]["status"] if items else "?"
            if status in ("READY", "FAILED"):
                parsed = status == "READY"
                check("resume parsed + embedded", parsed, status)
                break
        else:
            check("resume parsed + embedded", False, "timed out")

        print("\n=== 7. Question plan (LLM) ===")
        r = await c.post(f"{BASE}/interviews/{interview_id}/plan/generate", headers=H, json={})
        check("POST plan/generate", r.status_code in (200, 202), r.text[:200])

        print("  waiting for Nemotron...")
        plan = None
        for _ in range(50):
            await asyncio.sleep(3)
            r = await c.get(f"{BASE}/interviews/{interview_id}/plan", headers=H)
            if r.status_code == 200:
                plan = r.json()
                if plan["generation_status"] in ("READY", "FAILED"):
                    break
        ok = plan is not None and plan["generation_status"] == "READY"
        check(
            "plan generated",
            ok,
            f"{len(plan['questions'])} questions, {len(plan['criteria'])} criteria"
            if ok
            else str(plan.get("error", "timeout"))[:200] if plan else "no plan",
        )
        if ok:
            total = sum(float(x["weight"]) for x in plan["criteria"])
            check("rubric weights sum to 1.0", abs(total - 1.0) < 1e-6, f"{total}")
            r = await c.post(f"{BASE}/interviews/{interview_id}/plan/approve",
                             headers=H, json={"expected_version": plan["version"]})
            check("POST plan/approve", r.status_code == 200, r.text[:200])

        print("\n=== 8. A candidate must not see the plan ===")
        r = await c.get(f"{BASE}/interviews/{interview_id}/plan", headers=CH)
        check("candidate blocked from plan", r.status_code == 403, str(r.status_code))

        print("\n=== 9. Proctoring frame presign ===")
        r = await c.post(
            f"{BASE}/proctoring/frames/presign", headers=CH, json={"content_type": "image/jpeg"}
        )
        check("POST frames/presign", r.status_code == 200, r.text[:150])

        print("\n=== 10. Interview lifecycle + transcript ===")
        # Drive the session through the bus the way the voice pipeline does.
        import app.models  # noqa: F401
        from app.core import events
        from app.db.session import tenant_session
        from app.modules.interview import service as isvc
        from app.modules.interview import transcript as tmod

        # This process has no app lifespan, so nothing is subscribed to the bus.
        events.bus.clear()
        isvc.register()
        tmod.register()

        org_id = uuid.UUID(org["org_id"])
        iid = uuid.UUID(interview_id)

        events.publish(events.SessionStarted(org_id=org_id, interview_id=iid))
        await events.bus.drain()
        r = await c.get(f"{BASE}/interviews/{interview_id}", headers=H)
        check("SessionStarted -> IN_PROGRESS", r.json()["status"] == "IN_PROGRESS",
              r.json()["status"])

        turns = [
            ("INTERVIEWER", "Tell me about the sharding migration at Northwind.", 0, 5000),
            ("CANDIDATE",
             "We sharded on tenant id because every query already carried it. The cost was "
             "cross-tenant reporting, which we moved to a nightly rollup into ClickHouse. "
             "If I did it again I would have measured the reporting load first.", 6000, 32000),
            ("INTERVIEWER", "What did that migration cost you?", 33000, 36000),
            ("CANDIDATE",
             "About three months of two engineers, mostly dual-writing the write path and "
             "backfilling. We kept a kill switch on the read path the whole time, which we "
             "used once when the rollup lagged.", 37000, 60000),
            ("INTERVIEWER", "How do you approach code review?", 61000, 64000),
            ("CANDIDATE",
             "Um, I mean, I usually just look for obvious bugs. I do not really have a "
             "system for it.", 65000, 74000),
        ]
        async with tenant_session(org_id, "system", None) as s:
            for i, (spk, txt, a, b) in enumerate(turns):
                await tmod.record_turn(s, org_id=org_id, interview_id=iid, ordinal=i,
                                       speaker=spk, content=txt,
                                       started_offset_ms=a, ended_offset_ms=b)
        r = await c.get(f"{BASE}/interviews/{interview_id}/transcript", headers=H)
        check("transcript persisted", len(r.json()["turns"]) == 6, str(len(r.json()["turns"])))

        events.publish(events.SessionEnded(org_id=org_id, interview_id=iid,
                                           reason="completed", recording_key=None))
        await events.bus.drain()
        r = await c.get(f"{BASE}/interviews/{interview_id}", headers=H)
        check("SessionEnded -> COMPLETED", r.json()["status"] == "COMPLETED", r.json()["status"])

        print("\n=== 11. Post-interview pipeline (worker) ===")
        print("  waiting for scoring + reports...")
        score = None
        for _ in range(70):
            await asyncio.sleep(3)
            r = await c.get(f"{BASE}/interviews/{interview_id}/score", headers=H)
            if r.status_code == 200:
                score = r.json()
                if score["status"] in ("READY", "FAILED"):
                    break
        ok = score is not None and score["status"] == "READY"
        check("score produced", ok,
              f"overall={score['overall']} {score['recommendation']}" if ok
              else str(score)[:200] if score else "timeout")
        if ok:
            graded = [x for x in score["criteria"] if x["score"] is not None]
            check("criteria scored", len(graded) > 0, f"{len(graded)}/{len(score['criteria'])}")
            with_ev = [x for x in graded if x["evidence"]]
            check("evidence attached + verified", len(with_ev) > 0,
                  f"{len(with_ev)} criteria with quotes")
            check("coverage reported",
                  "rubric_coverage" in score["confidence_signals"],
                  str(score["confidence_signals"].get("rubric_coverage")))

        print("\n  waiting for reports...")
        rep = None
        for _ in range(60):
            await asyncio.sleep(3)
            r = await c.get(f"{BASE}/interviews/{interview_id}/report", headers=H)
            if r.status_code == 200:
                rep = r.json()
                if rep["status"] in ("READY", "FAILED"):
                    break
        ok = rep is not None and rep["status"] == "READY"
        check("recruiter report rendered", ok,
              str(rep.get("error"))[:200] if rep and not ok else "")
        if ok:
            r = await c.get(f"{BASE}/interviews/{interview_id}/report/download", headers=H)
            check("recruiter report download link", r.status_code == 200)
            if r.status_code == 200:
                d = await c.get(r.json()["download_url"])
                check("recruiter PDF fetched", d.status_code == 200 and d.content[:5] == b"%PDF-",
                      f"{len(d.content)} bytes")

        cand_rep = None
        for _ in range(40):
            r = await c.get(f"{BASE}/reports/me", headers=CH)
            if r.status_code == 200 and r.json()["status"] in ("READY", "FAILED"):
                cand_rep = r.json()
                break
            await asyncio.sleep(3)
        ok = cand_rep is not None and cand_rep["status"] == "READY"
        check("candidate feedback rendered", ok, str(cand_rep)[:150] if not ok else
              f"{len(cand_rep['strengths'])} strengths, {len(cand_rep['growth_areas'])} gaps")

        if ok:
            body = (await c.get(f"{BASE}/reports/me", headers=CH)).json()
            # On the FIELD NAMES, not the prose. Feedback legitimately contains
            # "Overall, you came across well" -- that is English, not a score.
            # The structural guarantee is that no field can carry one.
            fields = set(body)
            leaks = [f for f in fields if any(w in f.lower() for w in
                     ("score", "overall", "recommendation", "verdict", "rubric", "weight"))]
            check("candidate response has no score FIELD", not leaks, str(leaks))

            r = await c.get(f"{BASE}/reports/me/download", headers=CH)
            check("candidate download link", r.status_code == 200)
            if r.status_code == 200:
                d = await c.get(r.json()["download_url"])
                check("candidate PDF fetched",
                      d.status_code == 200 and d.content[:5] == b"%PDF-", f"{len(d.content)} bytes")
                import re

                from pypdf import PdfReader
                txt = "\n".join(p.extract_text() or ""
                                for p in PdfReader(io.BytesIO(d.content)).pages).lower()
                # Marks and bands only. A word like "rating" can appear in the
                # model's prose ("no rating is implied") without a score being
                # present; a "4/5" cannot.
                marks = re.findall(r"\d\s*(?:/|out of)\s*5", txt) + re.findall(r"\d+\s*%", txt)
                bands = [b for b in ("strong_hire", "no_hire", "borderline",
                                     "insufficient_evidence") if b in txt.replace(" ", "_")]
                check("candidate PDF has no mark or band", not marks and not bands,
                      f"marks={marks} bands={bands}")

        print("\n=== 12. Proctoring report ===")
        r = await c.get(f"{BASE}/interviews/{interview_id}/proctoring", headers=H)
        check("GET proctoring report", r.status_code == 200,
              (r.json().get("verdict") or {}).get("verdict", "no verdict"))

        print("\n=== 13. Candidate cannot reach recruiter surfaces ===")
        for label, path in [
            ("score", f"/interviews/{interview_id}/score"),
            ("report", f"/interviews/{interview_id}/report"),
            ("proctoring", f"/interviews/{interview_id}/proctoring"),
            ("jobs", "/jobs"),
        ]:
            r = await c.get(f"{BASE}{path}", headers=CH)
            check(f"candidate blocked from {label}", r.status_code == 403, str(r.status_code))

        print("\n=== 14. Cross-org isolation ===")
        slug2 = f"rival{uuid.uuid4().hex[:8]}"
        r = await c.post(f"{BASE}/auth/register-org", json={
            "org_name": "Rival", "slug": slug2,
            "admin_email": f"admin@{slug2}.example.com", "admin_password": PW})
        RH = {"Authorization": f"Bearer {r.json()['tokens']['access_token']}"}
        for label, path in [
            ("interview", f"/interviews/{interview_id}"),
            ("score", f"/interviews/{interview_id}/score"),
            ("report", f"/interviews/{interview_id}/report"),
            ("job", f"/jobs/{job_id}"),
        ]:
            r = await c.get(f"{BASE}{path}", headers=RH)
            check(f"other org 404 on {label}", r.status_code == 404, str(r.status_code))

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL GREEN")
    print(f"\norg slug: {SLUG}   interview: {interview_id}")


if __name__ == "__main__":
    started = time.time()
    asyncio.run(main())
    print(f"({time.time() - started:.0f}s)")
