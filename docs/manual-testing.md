# Manual testing

Two ways to drive the product by hand: a browser console at **`/dev`** for the
whole flow including the live voice call, and `scripts/check_nim.py` for when
the model endpoints are the thing you doubt.

## Start everything

```bash
# 1. Backing services (postgres + pgvector, redis, minio)
docker compose up -d

# One time only: create the database roles. CREATE ROLE is not a privilege
# Alembic should hold, so it does not live in a migration.
docker compose exec -T postgres psql -U postgres -d ai_interviewer \
  -v owner_pw="'owner_pw'" -v app_pw="'app_pw'" < scripts/bootstrap_db.sql

alembic upgrade head

# 2. The API
uvicorn app.main:app --reload

# 3. The worker, in another shell. Without it nothing after "upload resume"
#    happens: no parsing, no plan, no scoring, no reports.
celery -A app.workers.celery_app:celery_app worker --loglevel=info
```

Check it came up:

```bash
curl -s localhost:8000/health   # {"status":"ok"}
curl -s localhost:8000/ready    # every dependency, by name
```

`/ready` returning 503 tells you *which* dependency is down. Fix that before
anything else — a red storage check means presigned uploads will fail in a
confusing way three steps later.

## The console

Open **<http://localhost:8000/dev>**.

It is served by the API rather than opened as a file so it is same-origin — no
CORS entry to add, and no chance of pointing it at the wrong server. It is not
mounted at all when `ENVIRONMENT=production`.

The panels run top to bottom and each one leaves what the next one needs in
browser state. **Run full flow** does panels 1–3 plus a few proctoring events;
after that, generation takes 15–40 s, so press **Refresh** on panel 5.

| Panel | What to look for |
|---|---|
| 1 · Recruiter | Register mints a tenant and signs you in. The pill turns green. |
| 2 · Job | Creates job, description v1 and the proctoring policy in one go. |
| 3 · Invite | Prints the interview id and the candidate link. **Redeem** switches you to the candidate token — panels 4, 6 and 7 all need it. The header pill counts it down. |
| 4 · Resume | Needs the candidate token from panel 3. Presign → PUT straight to MinIO → complete. Press **Check** until `READY`. |
| 5 · Plan | Generation already started at invite time. Weight sum must read exactly `1.0000`. |
| 6 · Voice | Grants the mic, sends an SDP offer, plays the interviewer back. |
| 7 · Proctoring | Real tab-switches are reported while connected. |
| 8 · Terminate | Fires the whole post-interview chain. |
| 9 · Results | Score, proctoring verdict, and both PDFs. |

### Things worth trying to break

The interesting assertions are the negative ones.

- **Panel 7 → SECOND_SPEAKER (forged).** The ack count must not increase and
  the socket must stay open. That signal comes from the audio, not the browser;
  a candidate claiming it is fabricating evidence about themselves.
- **Panel 7 → switch tabs a few times** with the policy's blur limit at 2.
  Severity escalates INFO → WARN. Now reconnect and blur again: escalation must
  *not* reset, or a candidate clears their record by reopening the tab.
- **Panel 5 while signed in as the candidate.** 403. The plan is the answer key.
- **Panel 9 → Candidate feedback.** The console checks the response for
  score-bearing *fields* and prints `clean — no score fields`. Open the PDF and
  look for a mark (`4/5`), a percentage, or a band (`STRONG_HIRE`). There
  should be none.

  Searching the prose for the *word* "score" or "rating" will occasionally hit —
  the model writes ordinary English, and "Overall, you came across well" is a
  sentence, not a leak. The guarantee that matters is structural: no field on
  `candidate_reports`, on `CandidateView`, or on `CandidateFeedbackRead` can
  hold a score, and `candidate.generate` has no parameter to receive one.
- **Panel 8 twice.** The second terminate is a 409, not a second termination.
- **Panel 6 during shutdown.** Stop uvicorn with Ctrl-C while a call is live:
  new offers get 409 and `/ready` reports `draining`, while the call already
  running is allowed to finish.

### If the voice panel does not connect

In rough order of likelihood:

1. `pip install -e '.[voice]'` was never run. The API imports fine without it
   until the first offer.
2. `NVIDIA_API_KEY` is unset or spent. Run `python scripts/check_nim.py` — it
   probes the LLM, ASR and TTS functions separately and prints latency per
   service.
3. You are not on `localhost`. The transport configures host ICE candidates
   only, so it works browser-to-server on one machine or one LAN and fails
   across the internet. STUN/TURN is configuration, not code — see
   [ADR-0002](adr/0002-single-process-voice.md).
4. Magpie is cold. The first synthesis takes ~690 ms against ~70 ms warm; the
   session prewarms, but a cold NVCF function can still make the opening line
   feel late.

## The automated walkthrough

```bash
python scripts/e2e_walkthrough.py
```

Drives the same flow over real HTTP — register a tenant, create a job, invite,
redeem, upload a resume, generate a plan, run an interview, score it, download
both PDFs — and prints PASS/FAIL per step, so a failure points at one call
rather than at "the pipeline". It ends `ALL GREEN` or lists what broke.

Needs the API, a worker and the backing services up. It calls the real NVIDIA
endpoints, so a full run costs one plan generation plus one model call per
rubric criterion, and takes a few minutes.

Two steps reach past HTTP: the interview lifecycle is driven by publishing
`SessionStarted` / `SessionEnded` on the event bus, because the alternative is
a real WebRTC call with a real microphone. That part is what the console's
panel 6 is for.

## Driving it from a terminal instead

```bash
python scripts/seed_data.py            # org + admin + recruiter + job + candidate

TOKEN=$(curl -s localhost:8000/api/v1/auth/login \
  -H 'content-type: application/json' \
  -d '{"email":"admin@acme.example.com","password":"correct-horse-battery-staple"}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

curl -s localhost:8000/api/v1/jobs -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

Full request and response shapes are at <http://localhost:8000/docs>, generated
from the same Pydantic models the routes validate against.

## Reading the failures

| Symptom | Cause |
|---|---|
| `429` on register | Should not happen in development, where the cap is 500/hour. If it does, you have `REGISTER_MAX_ATTEMPTS` set in `.env`, or `ENVIRONMENT` is not `development`. To clear the counters: `docker exec aii-redis redis-cli --scan --pattern 'rl:*' \| xargs docker exec -i aii-redis redis-cli DEL` |
| `401 unauthenticated` on a candidate panel | No candidate token. Panel 3: **Create invite**, then **Redeem**. The console now renews it automatically when it lapses, but it cannot invent one before an invite exists. |
| Resume stuck `UPLOADED` | No worker running. |
| Plan stuck `GENERATING` | Model call in flight; 15–40 s, longer if it needs a repair turn. A crashed worker unsticks after 5 minutes. |
| Score `INSUFFICIENT_EVIDENCE` | Fewer than half the rubric's weight could be graded. Not a bug — check the transcript actually has candidate turns. |
| Score 404 after terminating | The chain runs `finalize → transcript → signals ∥ frames → score → verdict → reports`. Watch the worker log. |
| Report 404 | Same chain, two steps later. |
| `NotFoundError` in the worker log | Stale tasks from an earlier run against rows that no longer exist. Drain with `docker exec aii-redis redis-cli -n 1 FLUSHDB`. |

## What is not covered

- **The candidate never gets a real second speaker.** `SECOND_SPEAKER` comes
  from sortformer diarization on the live ASR stream, so testing it needs two
  people audible in one room.
- **Webcam frames** are presigned and uploaded by the console, but the offline
  vision pass only runs frames that were actually reported over the socket.
- **The full Celery chord** is exercised by terminating an interview, but a
  single-worker setup serialises what production runs concurrently.
