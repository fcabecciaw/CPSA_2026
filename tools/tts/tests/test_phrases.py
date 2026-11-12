"""Extraction of the dashboard's fixed phrases from the TypeScript sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpsa_tts.config import repo_root
from cpsa_tts.phrases import (
    SLOT_RE,
    collect,
    parse_facts,
    parse_poems,
    parse_rulebook,
    write_phrases,
)

SRC = repo_root() / "web" / "src" / "cognitive"


def test_collect_reads_the_three_sources() -> None:
    document = collect(repo_root())
    counts = document.counts()
    # Pinned to the base commit (5d4d359); the test fails loudly when the
    # dashboard adds phrases, which is the signal to re-run `pregenerate`.
    assert counts["rulebook"] == 97
    assert counts["fact"] == 40
    assert counts["poem"] == 20
    assert counts["total"] == 157


def test_no_fixed_phrase_has_a_slot() -> None:
    document = collect(repo_root())
    assert [phrase.text for phrase in document.phrases if SLOT_RE.search(phrase.text)] == []


def test_object_class_lines_are_excluded_because_they_have_slots() -> None:
    texts = {phrase.text for phrase in parse_rulebook(SRC / "rulebook.ts")}
    assert "Ecco {oggetto}: si pedala?" not in texts
    assert "Ciao! Ti vedo." in texts


def test_rulebook_phrases_are_decoded_and_short() -> None:
    phrases = parse_rulebook(SRC / "rulebook.ts")
    assert all("\\" not in phrase.text for phrase in phrases)
    # F-E4 #17: the dashboard never speaks more than ~12 words per sentence.
    assert max(len(phrase.text.split()) for phrase in phrases) <= 20


def test_facts_keep_their_ids_and_text() -> None:
    facts = parse_facts(SRC / "content" / "facts.ts")
    by_id = {phrase.id: phrase for phrase in facts}
    assert "fact.demo-60fps" in by_id
    assert by_id["fact.demo-60fps"].text.startswith("Guardo il mondo sessanta volte al secondo")


def test_poems_join_lines_with_one_space() -> None:
    poems = parse_poems(SRC / "content" / "poems.ts")
    first = next(phrase for phrase in poems if phrase.id == "poem.p-sessanta")
    assert first.text == "Sessanta sguardi in un secondo, e ancora non ho capito il mondo."
    assert "\n" not in first.text


def test_duplicates_are_removed_but_order_is_kept(tmp_path: Path) -> None:
    source = tmp_path / "rulebook.ts"
    source.write_text(
        'a: { phrases: ["Ciao!", "Ciao!", "A presto!"] },\nb: { phrases: ["Doppio: {n}"] },\n',
        encoding="utf-8",
    )
    phrases = parse_rulebook(source)
    assert [phrase.text for phrase in phrases] == ["Ciao!", "A presto!"]


def test_write_phrases_is_utf8_and_json_readable(tmp_path: Path) -> None:
    document = collect(repo_root())
    path = write_phrases(tmp_path / "build" / "phrases.json", document)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["counts"]["total"] == 157
    assert "e ancora non ho capito" in path.read_text(encoding="utf-8")
    assert payload["phrases"][0]["source"] == "rulebook"


def test_collect_fails_loudly_without_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no phrases"):
        collect(tmp_path)
