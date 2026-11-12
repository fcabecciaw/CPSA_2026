# ARI composes its lines with a small local LLM — Implementation Plan

**Status:** DRAFT for review (2026-11-12) — after the open day; tonight only the deterministic speed-ups (T1) run.
**Goal:** ARI sounds like a person: the rule engine still decides *when* to speak and *what about*, a small LLM on the
laptop composes the actual sentence from approved examples and the recent context, and the voice starts within about
a second.
**Spec (user, 2026-11-12, verbatim):** "I think that an llm, even if small, may compose pre-written phrases better than
an handmade speaker." Earlier: "We should make it so it speaks fluidly, composing phrases etc." / "It's important that
the phrases do not assume the gender. So it's probably worth it to always speak in neutral".
**Non-goals (this plan):** speech input / two-way conversation (later phase L4); cloud APIs (no API key available).
**Constraints:** runs on the laptop (Ubuntu 24.04, RTX PRO 500 Blackwell 6 GB, shared with XTTS ≈ 2.6 GB); offline;
child-safe, gender-neutral Italian; never invents facts; never claims to recognise a person; the board is unchanged.

## Approach — "constrained composer"

```
board -> dashboard (browser): perception -> events -> rule engine (when + intent + slots)   [unchanged]
                                                   |
                                                   v
  composer (laptop, 127.0.0.1:8766): prompt = persona + intent + slots + 3-5 APPROVED example lines for that rule
                                     + last 4 utterances + scene state  -> 1 sentence (<= 14 words)
                                                   |
                                     validator (browser): length, banned/gendered words, required slot words,
                                     no digits/URLs/emoji, not a repeat  --fail/timeout 700 ms--> approved line
                                                   |
                                     voice: cached approved lines play at once; composed lines use XTTS streaming
```

- **LLM runtime:** llama.cpp server (OpenAI-compatible HTTP), GGUF Q4_K_M on the GPU, grammar-constrained output (GBNF:
  one line, Italian letters/punctuation only, max ~120 chars), temperature ~0.7, max ~32 tokens; or Ollama if simpler.
- **Candidates (to evaluate, L0):** Velvet 2B (Almawave, Italian-native, Apache-2.0, on Ollama), Gemma 3 4B / Gemma 3n
  E2B, Qwen 2.5/3 3–4B instruct, Minerva 3B (Sapienza) — pick by validity rate, Italian naturalness (user's ear),
  latency (target p95 <= 600 ms for 20 tokens) and VRAM next to XTTS.
- **Prompting:** persona sheet (ARI, friendly, witty, neutral Italian, short), the rule's intent in plain words
  ("greet a group of 3 that just arrived", "react to the 3rd phone today"), the rule's approved lines as few-shot
  examples, a short memory of what was said, forbidden list (gendered forms, identity claims, body/age/looks).
- **Facts stay curated:** the LLM may only add a short lead-in ("Lo sapevi che…") to a curated fact, never invent one.
- **Validator** reuses the phrase test lists (gendered words, recognition phrases) at runtime; any failure or timeout
  falls back to the approved line, so the demo never depends on the LLM.
- **Voice:** XTTS streaming inference for composed lines (first audio chunk ~0.3 s), plus prefetch while the previous
  line is still playing; approved lines stay pre-generated.
- **Switch:** `composer: off | on` in the dashboard debug panel; default off until the user approves the outputs.

## Verification contract

- **AC-L1** eval harness: 60 intents x each candidate model -> validity %, p50/p95 latency, VRAM, and a listening/reading
  sheet for the user; the chosen model has >= 95% valid outputs and p95 <= 600 ms with XTTS loaded.
- **AC-L2** composer service: health, timeout, grammar constraint, deterministic tests with a fake model.
- **AC-L3** validator + fallback: unit tests for every forbidden category; a broken/slow composer never produces silence
  or a forbidden line (fault-injection tests).
- **AC-L4** end to end on the laptop: event -> first audio <= 1.2 s p95 for composed lines, <= 0.3 s for cached lines.
- **AC-L5** user review of 100 composed lines before enabling it at an event.

## Tasks (after the open day)

- **L0** eval harness + model choice (1–2 h).
- **L1** composer service (`tools/composer/`, llama.cpp/Ollama client, prompts, systemd user unit) (half day).
- **L2** web integration: composer client, validator, fallback, debug switch, memory of recent lines (half day).
- **L3** XTTS streaming in `tools/tts` + browser chunked playback (half day).
- **L4** (separate plan) two-way conversation: faster-whisper Italian ASR with push-to-talk at the stand, then LLM reply.

## Open decisions / risks

- Model choice by ear (Italian quality) — needs the user.
- GPU memory: XTTS + 4B Q4 is ~5 GB of 6; fallback = 2B model or LLM on CPU (slower).
- Safety at a public event: approved-only mode remains one click away.
