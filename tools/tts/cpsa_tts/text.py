"""Text normalization before synthesis: say the words, never the punctuation.

The dashboard writes for the eye ("Sembra un'auto… dico bene?", "«Ciao»",
"2 × 3"); a neural voice reads some of those marks aloud or turns them into odd
pauses. :func:`normalize` is the single place that turns a dashboard sentence
into what the engine speaks. ``POST /tts``, the pregenerate job, the CLI
``pregenerate``/``samples``/``synthesize`` and the cache key all use it, so a
sentence pregenerated offline is the very same cache entry the dashboard asks
for.

Rules, in order:

- ellipsis (``…`` or ``...``), colon and semicolon become a comma pause;
- ``×`` becomes the word " per ";
- double quotes, guillemets, parentheses, brackets, asterisks, underscores and
  emoji are removed; the apostrophe stays (``un'auto``, ``c'è``);
- repeated ``!``/``?`` collapse to one mark (``?`` wins when present);
- spaces before punctuation, doubled commas and repeated spaces are cleaned;
- for XTTS only, the final ``.`` is dropped (it is sometimes voiced), ``?`` and
  ``!`` are kept for the intonation.
"""

from __future__ import annotations

import re

#: Engines that read a final full stop aloud.
DROP_FINAL_PERIOD = frozenset({"xtts"})

_ELLIPSIS_RE = re.compile(r"\s*(?:…|\.{2,})\s*")
_PAUSE_RE = re.compile(r"\s*[:;]\s*")
_TIMES_RE = re.compile(r"\s*×\s*")
#: Marks that are never spoken: quotes, guillemets, brackets, emphasis.
_DROP_RE = re.compile(r"[\"“”„«»()\[\]{}*_~`#]")
#: Emoji and pictographs, variation selectors and zero-width joiners.
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoji, symbols & pictographs ext.
    "\u2600-\u27bf"  # misc symbols, dingbats
    "\u2b00-\u2bff"  # arrows, stars
    "\ufe0e\ufe0f\u200d\u20e3"  # variation selectors, ZWJ, keycap
    "]"
)
_MARKS_RE = re.compile(r"[!?][!?]+")
_SPACE_BEFORE_RE = re.compile(r"\s+([,.!?])")
#: Doubled commas left by removed marks; "4,1" (Italian decimals) is untouched.
_COMMAS_RE = re.compile(r",(?:\s*,)+")
#: A comma right before a stop is a leftover of "…?" or ":." – keep the stop.
_COMMA_STOP_RE = re.compile(r",\s*([.!?])")
_SPACES_RE = re.compile(r"\s+")


def _collapse_marks(match: re.Match[str]) -> str:
    return "?" if "?" in match.group(0) else "!"


def normalize(text: str, engine: str | None = None) -> str:
    """The sentence the engine actually speaks (and the cache key's text)."""
    spoken = _ELLIPSIS_RE.sub(", ", text)
    spoken = _PAUSE_RE.sub(", ", spoken)
    spoken = _TIMES_RE.sub(" per ", spoken)
    spoken = _DROP_RE.sub("", spoken)
    spoken = _EMOJI_RE.sub("", spoken)
    spoken = _MARKS_RE.sub(_collapse_marks, spoken)
    spoken = _SPACE_BEFORE_RE.sub(r"\1", spoken)
    spoken = _COMMAS_RE.sub(",", spoken)
    spoken = _COMMA_STOP_RE.sub(r"\1", spoken)
    spoken = _SPACES_RE.sub(" ", spoken).strip()
    # A sentence that started with a pause mark ("… allora?") starts clean.
    spoken = spoken.lstrip(", ").rstrip(", ")
    if engine in DROP_FINAL_PERIOD:
        spoken = spoken.rstrip(".").rstrip()
    return spoken


#: A sentence ends at ``.``/``!``/``?`` followed by a space ("10.000" stays whole).
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    """Split a normalized utterance into sentences, each keeping its mark."""
    return [part.strip() for part in _SENTENCE_END_RE.split(text) if part.strip(" .!?")]
