# 1. FastAPI over Fastify

Status: accepted

## Context

The reference architecture specified Fastify. The voice pipeline is Python
regardless -- pipecat, Riva's client, Silero, librosa and the NVIDIA NIM SDKs
have no JavaScript equivalent worth using.

## Decision

FastAPI for the whole backend.

## Consequences

The tradeoff the original architecture named -- "it splits the codebase into two
languages and we lose shared types with the frontend" -- is inverted here,
because the voice pod is Python either way. Choosing Fastify would have meant
running both languages *and* putting a network hop between the interview
orchestration and the pipeline that needs its question plan on the turn budget.

What is genuinely lost: shared TypeScript types with the frontend. The mitigation
is the OpenAPI schema FastAPI generates, which a client generator can consume.

Celery replaces BullMQ for the same reason: the workers run Python code
(resume parsing, embeddings, librosa, WeasyPrint) and a Node queue would need a
Python worker anyway.
