# API Reference

Base path `/api/v1`. Interactive docs at `/docs` — disabled in production,
where an OpenAPI schema is an attack-surface map.

This page is about **who may call what and why**. Request and response shapes
live in the OpenAPI schema, which is generated from the same Pydantic models
the routes validate against and therefore cannot drift.

## Two kinds of caller

| | Recruiter / admin | Candidate |
|---|---|---|
| Credential | email + password | magic link, no password |
| Session token | `access` (30 min) | `interview` (10 min) |
| Obtained via | `POST /auth/login` | `POST /auth/invites/redeem` |
| Header | `Authorization: Bearer <token>` | same |

The two token types are signed with **different derived keys**, so presenting
one where the other belongs fails at signature verification rather than at a
claim check. A recruiter token on a candidate route is `403`, not a partial
success.

`org_id` is never accepted from a client — not in a path, query, body or
header. It comes from the token's signed claims and drives Postgres RLS.

## Status codes

| Code | Meaning here |
|---|---|
| `401` | no token, expired, or wrong signature |
| `403` | valid token, wrong actor kind or role |
| `404` | not found **or** belongs to another org — the two are indistinguishable by design |
| `409` | state conflict (interview still live, plan frozen, version mismatch) |
| `410` | invite link no longer valid |
| `422` | request body failed validation |
| `429` | rate limited (login, registration, proctoring socket) |
| `503` | dependency down or instance draining |

A cross-tenant read is a `404`, never a `403`. A `403` would confirm the
resource exists.

---

## Auth

| Method | Path | Caller |
|---|---|---|
| POST | `/auth/register-org` | public, rate limited |
| POST | `/auth/login` | public, rate limited |
| POST | `/auth/refresh` | refresh token |
| POST | `/auth/logout` | refresh token |
| POST | `/auth/logout-all` | recruiter |
| GET | `/auth/me` | any |
| POST | `/auth/invites` | recruiter |
| POST | `/auth/invites/redeem` | public |

`/auth/logout` is `204` even for an already-dead token — a `401` would tell an
attacker which tokens are still live.

Refresh rotates: the old token is consumed and a replay of it kills the whole
family, logging out both the attacker and the legitimate user. That is the
intended outcome; one of the two holds a stolen token and we cannot tell which.

`/auth/invites` returns the invite token in the response **and** emails it. The
email is best-effort — a dead mail server does not roll back the invite, and the
link is recoverable from the response.

## Users, candidates, jobs

| Method | Path | Caller |
|---|---|---|
| GET/POST | `/users`, `/users/{id}` | admin (recruiters may read themselves) |
| PATCH/DELETE | `/users/{id}` | admin |
| GET/POST/PATCH/DELETE | `/candidates`, `/candidates/{id}` | recruiter |
| GET/POST/PATCH/DELETE | `/jobs`, `/jobs/{id}` | recruiter |
| GET/POST | `/jobs/{id}/descriptions` | recruiter |
| POST | `/jobs/{id}/descriptions/{id}/activate` | recruiter |

Descriptions are versioned and immutable; "editing" adds a version and
activates it, so a plan generated last month still names the text it was
generated from.

Deactivating the last admin is a `409`. An org with no admin cannot be
recovered through the API.

## Resumes — candidate-uploaded

| Method | Path | Caller |
|---|---|---|
| POST | `/candidates/me/resume/presign` | **candidate** |
| POST | `/candidates/me/resume/{id}/complete` | **candidate** |
| GET | `/candidates/me/resume` | candidate |
| GET | `/candidates/{id}/resumes` | recruiter |
| GET | `/candidates/{id}/resumes/{id}/download` | recruiter |

The candidate is the only party holding the file, so they upload it — the one
place a candidate writes to the database. The S3 key is server-chosen and
org-prefixed: a client that could name the key could overwrite another
candidate's file.

`/complete` is idempotent on the resulting status, so a double-tap enqueues the
parse pipeline once.

## Interviews

| Method | Path | Caller |
|---|---|---|
| GET/POST | `/interviews` | recruiter |
| GET | `/interviews/{id}` | recruiter |
| GET | `/interviews/{id}/transcript` | recruiter |
| POST | `/interviews/{id}/terminate` | recruiter |
| GET | `/interviews/me` | **candidate** |

No route sets `IN_PROGRESS` or `COMPLETED`. Those transitions belong to the
voice session and arrive over the event bus; a recruiter can only *terminate*,
which is a deliberate act rather than a lifecycle step.

`/interviews/me` resolves from the token's own claim — a candidate cannot ask
about someone else's interview because there is nowhere to put another id.

## Question plans

| Method | Path | Caller |
|---|---|---|
| GET | `/interviews/{id}/plan` | recruiter |
| POST | `/interviews/{id}/plan/generate` | recruiter |
| PUT | `/interviews/{id}/plan/questions` | recruiter |
| PUT | `/interviews/{id}/plan/criteria` | recruiter |
| POST | `/interviews/{id}/plan/approve` | recruiter |

**No candidate route exists, and RLS enforces it** — the plan is the answer key.

Generation starts automatically at **invite time**, not when a recruiter asks.
A candidate can redeem a link within seconds while generation takes tens of
seconds, so deferring it to a click is how someone gets interviewed with no plan
at all. `POST /plan/generate` is for regeneration; it returns
`skipped: already generating` if one is already in flight, so the two paths
cannot race each other into a half-written rubric.

Edits carry the version the client read; a mismatch is `409` so two recruiters
cannot silently overwrite each other. The plan freezes when the interview
starts and is immutable after: an interview must be scorable against the exact
questions and weights it was conducted under.

Criterion weights must sum to 1.0. The generator rescales proportionally rather
than rejecting — measured behaviour is that Nemotron returns 1.05 and repeats
it when shown the error, and the model's real contribution is the *relative*
importance.

## Voice

| Method | Path | Caller |
|---|---|---|
| POST | `/webrtc/offer` | **candidate** |

Accepts `INVITED` (first join) and `IN_PROGRESS` (reconnect — the invite is
multi-use for exactly this reason). Terminal is `409`. During shutdown this
returns `409` with a retry hint rather than building a pipeline against a pool
that is about to close.

The system prompt is assembled server-side and never appears in any response.

## Proctoring

| Method | Path | Caller |
|---|---|---|
| GET/PUT | `/jobs/{id}/proctoring-policy` | recruiter |
| GET | `/interviews/{id}/proctoring` | recruiter |
| POST | `/proctoring/frames/presign` | **candidate** |
| WS | `/proctoring/ws?token=…` | **candidate** |

A candidate cannot read the policy — knowing the blur limit is knowing exactly
how much you can get away with.

The WebSocket is the one place in this API where the caller *is* the person
being assessed, so nothing from it is trusted: the browser may say *what*
happened, never how serious it is, when it happened, or which interview it
belongs to. Severity comes from the rules, the timestamp from the server clock,
the interview from the token. `SECOND_SPEAKER`, `MULTIPLE_FACES`, `FACE_ABSENT`
and `ANOMALOUS_SILENCE` are server-derived and rejected if claimed.

Every message is acknowledged with `{"accepted": n}` — a running count, never a
reason. The forger already knows what they sent; telling them which rule caught
it would be a tuning loop.

The token is a query parameter because a browser WebSocket cannot set an
`Authorization` header. It is verified *before* the connection is accepted.

Escalation counters are primed from earlier events, so reconnecting does not
reset them.

## Scoring

| Method | Path | Caller |
|---|---|---|
| GET | `/interviews/{id}/score` | recruiter |
| POST | `/interviews/{id}/score/rescore` | recruiter |

**No candidate route, and `scores` is USER_ONLY at the RLS layer** — a future
endpoint that forgets the role check still returns nothing.

The score row exists in `PENDING` before there is anything in it, so a
recruiter opening the page seconds after the call ends sees "in progress"
rather than a `404` that reads like scoring never started.

`overall` is `null` when rubric coverage falls below 50%; the recommendation is
then `INSUFFICIENT_EVIDENCE`, which is **not** `NO_HIRE`. Each criterion
carries its verified evidence with the offset in the recording.

`confidence_signals` are observations — pitch, pauses, fillers, plus rubric
coverage. Nothing in there was multiplied into the score.

Rescoring a live interview is `409`. The previous score stays readable until
the new one replaces it.

## Reports

| Method | Path | Caller |
|---|---|---|
| GET | `/interviews/{id}/report` | recruiter |
| GET | `/interviews/{id}/report/download` | recruiter |
| POST | `/interviews/{id}/report/regenerate` | recruiter |
| GET | `/reports/me` | **candidate** |
| GET | `/reports/me/download` | **candidate** |

Two audiences, separated at every layer: separate tables, RLS policies,
builders, templates and schemas. There is no endpoint that takes an audience
parameter.

The candidate response contains no score, band, recommendation, weight or
verdict — and cannot, because `candidate_reports` has no such column and
`CandidateFeedbackRead` has no such field.

Downloads return a presigned URL, never the S3 key: a bucket name is guessable
and a key is structured, so returning both is most of the way to handing out
the object. Each re-render writes a new key, so a URL already in someone's hand
keeps resolving to the version they were reading.

`regenerate` re-renders **both** audiences. Letting them drift to different
vintages is how a candidate receives feedback that contradicts the report the
recruiter is reading.

## Ops

Unversioned and unauthenticated — a probe is issued by an orchestrator that has
no credentials.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness; `200` while the process runs |
| GET | `/ready` | readiness; `503` if a dependency is down or draining |
| GET | `/metrics` | uptime, active voice sessions, pending bus handlers |

`/ready` checks Postgres, Redis and S3 concurrently with a per-check timeout,
and reports each by name so one probe tells an operator everything that is
wrong.
