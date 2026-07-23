# 2. Single process, voice in-process

Status: accepted (with known risk)

## Context

The voice pipeline holds a live call: ASR, LLM and TTS streaming on a turn
budget, with a real person waiting. Every other component in this system is
disposable -- kill an API pod mid-request and the client retries. Kill the pod
holding a voice session and someone's interview ends.

## Decision

Run the pipeline in the API process. `app/modules/voice/session_manager.py` is
the only public surface; nothing outside the module imports its internals, and
everything it tells the rest of the system goes over `app/core/events.py`.

## Consequences

Accepted now:

- The API process becomes stateful. It needs sticky sessions, slow drain on
  deploy, and a shutdown path that ends live sessions before the event bus is
  drained (`app/main.py` lifespan does this).

  The drain is now explicit: the lifespan sets `app/api/ops.py:set_draining`
  before it closes anything, `POST /webrtc/offer` returns 409 from that moment,
  and `/ready` reports 503 so the load balancer stops routing. Sessions already
  running are left alone to finish — each is a real person mid-sentence, and
  the point of draining is to stop *new* work, not to cut off current work.
- A deploy interrupts live interviews. The mitigations are the per-turn Redis
  checkpoint and the multi-use invite: a candidate reconnects and resumes at
  the turn they left off, rather than starting over.

Because the module boundary is the event bus rather than function calls,
extracting the pipeline into its own process later is a deployment change, not
a rewrite. The subscribers in `interview/service.py` and `interview/transcript.py`
do not know or care which process published the event.

## Deployment debt, stated plainly

- **WebRTC needs STUN and TURN.** `app/modules/voice/transport.py` configures
  host ICE candidates only, which works browser-to-server on one machine or one
  LAN and fails across the internet. STUN handles most NAT traversal; TURN is
  required for the roughly one candidate in ten behind a symmetric NAT or a
  restrictive firewall. Both are configuration on `SmallWebRTCConnection`
  (`WEBRTC_STUN_URLS` already exists), not code changes.
- **One process, one machine.** `session_manager._sessions` is an in-process
  dict, so a second replica cannot see the first's sessions. Horizontal scaling
  needs sticky routing by interview id before it needs anything else.
