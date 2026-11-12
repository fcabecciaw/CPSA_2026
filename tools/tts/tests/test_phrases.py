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
    parse_questions,
    parse_rulebook,
    parse_scene,
    write_phrases,
)

SRC = repo_root() / "web" / "src" / "cognitive"


def test_collect_reads_the_rulebook_the_facts_the_scenes_and_the_questions() -> None:
    document = collect(repo_root())
    counts = document.counts()
    # Pinned to the current dashboard; the test fails loudly when it adds
    # phrases, the signal to re-run `pregenerate`. Scene lines that repeat a
    # rulebook phrase are kept once, under the rulebook.
    assert counts["rulebook"] == 100
    assert counts["fact"] == 40
    assert counts["scene"] == 57
    assert counts["question"] == 36
    assert "poem" not in counts
    assert counts["total"] == 233
    texts = [phrase.text for phrase in document.phrases]
    assert len(texts) == len(set(texts))


def test_no_fixed_phrase_has_a_slot() -> None:
    document = collect(repo_root())
    assert [phrase.text for phrase in document.phrases if SLOT_RE.search(phrase.text)] == []


def test_object_class_lines_are_excluded_because_they_have_slots() -> None:
    texts = {phrase.text for phrase in parse_rulebook(SRC / "rulebook.ts")}
    assert "Vedo {oggetto}. Caffè o tè?" not in texts
    assert "Ciao! Ti vedo benissimo." in texts


def test_rulebook_phrases_are_decoded_and_short() -> None:
    phrases = parse_rulebook(SRC / "rulebook.ts")
    assert all("\\" not in phrase.text for phrase in phrases)
    # F-E4 #17: the dashboard never speaks more than ~12 words per sentence.
    assert max(len(phrase.text.split()) for phrase in phrases) <= 20


def test_scene_lines_are_every_slot_free_scene_phrase() -> None:
    texts = [phrase.text for phrase in parse_scene(SRC / "scene.ts")]
    # 67 slot-free lines of `SCENE_PHRASES` (greetings, intro, prompts,
    # invitations, goodbyes, reactions, continuity); none repeats.
    assert len(texts) == 67
    assert texts[0] == "Ciao! Che piacere vederti."
    assert "Piacere, sono ARI! Riconosco oggetti, gesti e sorrisi." in texts
    assert "Mouse e tastiera: qui si programma!" in texts
    # The `{n}` greeting and the `{oggetto}` follow-ups are rendered per event.
    assert "Ciao a tutti! Siete in {n}." not in texts
    assert not any(SLOT_RE.search(text) for text in texts)
    # The CONTINUITY keys are object pairs, not lines.
    assert "tazza+libro" not in texts
    # Comments in the tables are not lines either.
    assert not any("F-E4" in text for text in texts)


def test_scene_parser_skips_keys_comments_and_slots(tmp_path: Path) -> None:
    source = tmp_path / "scene.ts"
    source.write_text(
        'const OTHER = ["Fuori"];\n'
        "const GREETINGS = {\n"
        '  /** "Citazione" in a comment. */\n'
        '  single: ["Ciao!", "Ciao a {n}!"],\n'
        "};\n"
        '// "riga" commentata\n'
        'const CONTINUITY = { "a+b": ["Prima a, poi b."] };\n'
        'export const SCENE_PHRASES = { X: ["Dopo"] };\n',
        encoding="utf-8",
    )
    assert [phrase.text for phrase in parse_scene(source)] == ["Ciao!", "Prima a, poi b."]


def test_questions_bring_their_yes_and_no_answers() -> None:
    phrases = parse_questions(SRC / "content" / "questions.ts")
    by_id = {phrase.id: phrase for phrase in phrases}
    assert len(phrases) == 36
    assert by_id["question.q-tech"].text == "Ti piace la tecnologia?"
    assert by_id["question.q-tech"].kind == "question"
    assert by_id["question.q-tech.yes.0"].text == "Lo sapevo!"
    assert by_id["question.q-maths.no.1"].text == "Peccato, per me tutto è un numero."
    assert by_id["question.q-maths.no.1"].kind == "answer"
    assert sum(phrase.kind == "question" for phrase in phrases) == 7


def test_facts_keep_their_ids_and_text() -> None:
    facts = parse_facts(SRC / "content" / "facts.ts")
    by_id = {phrase.id: phrase for phrase in facts}
    assert "fact.demo-60fps" in by_id
    assert by_id["fact.demo-60fps"].text.startswith("Guardo il mondo sessanta volte al secondo")


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
    assert payload["counts"]["total"] == 233
    assert "nuraghi" in path.read_text(encoding="utf-8")
    assert payload["phrases"][0]["source"] == "rulebook"


def test_collect_fails_loudly_without_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no phrases"):
        collect(tmp_path)
