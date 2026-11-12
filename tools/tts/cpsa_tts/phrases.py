"""Extract the dashboard's fixed Italian sentences from the TypeScript sources.

The dashboard is the source of truth: ``web/src/cognitive/rulebook.ts`` (every
phrase without a ``{oggetto}``/``{n}``/``{ora}`` slot) and ``content/facts.ts``
(all facts). The files are read as text and
parsed with narrow regular expressions on purpose: this package must not depend
on Node, Vite or the web toolchain, and it must not import the dashboard code.

The texts are used verbatim as cache keys, so the parser preserves the strings
exactly (TypeScript escapes are decoded with ``json.loads``).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: A phrase with a slot is rendered per event and is never fixed. Besides the
#: documented ``{oggetto}``/``{n}``/``{ora}``, the rulebook also renders
#: ``{minuti}``, ``{dpu}``, ``{totale}`` and ``{fps}``, so any brace counts.
SLOT_RE = re.compile(r"\{\w+\}")
STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
PHRASES_ARRAY_RE = re.compile(r"phrases:\s*\[(.*?)\]", re.S)
FACT_RE = re.compile(
    r'\{\s*id:\s*"(?P<id>[^"]+)",\s*topic:\s*"[^"]+",\s*text:\s*"(?P<text>(?:[^"\\]|\\.)*)"\s*\}'
)


def web_source_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "web" / "src"


@dataclass(frozen=True, slots=True)
class Phrase:
    """One fixed sentence of the dashboard."""

    id: str
    text: str
    source: str  # "rulebook" | "fact"
    kind: str  # "rule" | "fact"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PhrasesDocument:
    """The exported ``phrases.json`` plus its counts."""

    phrases: tuple[Phrase, ...]
    generated_at: str

    def counts(self) -> dict[str, int]:
        counts = {"rulebook": 0, "fact": 0}
        for phrase in self.phrases:
            counts[phrase.source] = counts.get(phrase.source, 0) + 1
        counts["total"] = len(self.phrases)
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "counts": self.counts(),
            # The dashboard text is the key: keep the file readable and diffable.
            "phrases": [phrase.to_dict() for phrase in self.phrases],
        }


def parse_rulebook(path: Path) -> list[Phrase]:
    """Every literal ``phrases: [...]`` array without slots, in file order."""
    source = Path(path).read_text(encoding="utf-8")
    phrases: list[Phrase] = []
    for block in PHRASES_ARRAY_RE.finditer(source):
        for match in STRING_RE.finditer(block.group(1)):
            text = _decode(match.group(1))
            if SLOT_RE.search(text):
                continue
            phrases.append(
                Phrase(
                    id=f"rulebook.{len(phrases):03d}",
                    text=text,
                    source="rulebook",
                    kind="rule",
                )
            )
    return _unique(phrases)


def parse_facts(path: Path) -> list[Phrase]:
    """Every ``Fact.text`` of ``content/facts.ts``."""
    source = Path(path).read_text(encoding="utf-8")
    phrases = [
        Phrase(
            id=f"fact.{match.group('id')}",
            text=_decode(match.group("text")),
            source="fact",
            kind="fact",
        )
        for match in FACT_RE.finditer(source)
    ]
    return _unique(phrases)


def collect(repo_root: Path) -> PhrasesDocument:
    """All fixed phrases of the dashboard, in a stable order."""
    sources = web_source_dir(repo_root)
    paths = (
        sources / "cognitive" / "rulebook.ts",
        sources / "cognitive" / "content" / "facts.ts",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"no phrases found: missing {', '.join(missing)}")
    phrases = [
        *parse_rulebook(paths[0]),
        *parse_facts(paths[1]),
    ]
    if not phrases:
        raise ValueError(f"no phrases found under {sources}")
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    return PhrasesDocument(phrases=tuple(phrases), generated_at=generated_at)


def write_phrases(path: Path, document: PhrasesDocument) -> Path:
    """Write ``phrases.json`` (UTF-8, Italian characters unescaped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return path


def _decode(literal: str) -> str:
    """Decode the body of a double-quoted TypeScript string literal."""
    return json.loads(f'"{literal}"')


def _unique(phrases: list[Phrase]) -> list[Phrase]:
    """Drop repeated sentences; the order of first appearance is kept."""
    seen: set[str] = set()
    unique: list[Phrase] = []
    for phrase in phrases:
        if phrase.text in seen:
            continue
        seen.add(phrase.text)
        unique.append(phrase)
    return unique
