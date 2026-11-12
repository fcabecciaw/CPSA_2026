"""Text normalization before synthesis (T1): no punctuation is read aloud."""

from __future__ import annotations

import pytest

from cpsa_tts.text import normalize, sentences


@pytest.mark.parametrize(
    ("text", "xtts", "other"),
    [
        # Real dashboard lines (rulebook, scene, facts).
        ("Sembra un'auto… dico bene?", "Sembra un'auto, dico bene?", "Sembra un'auto, dico bene?"),
        ("Ciao a tutti! Siete in 3.", "Ciao a tutti! Siete in 3", "Ciao a tutti! Siete in 3."),
        (
            "Psst! Sì, dico proprio a te.",
            "Psst! Sì, dico proprio a te",
            "Psst! Sì, dico proprio a te.",
        ),
        (
            "Mi sembra di vedere una forchetta… possibile?",
            "Mi sembra di vedere una forchetta, possibile?",
            "Mi sembra di vedere una forchetta, possibile?",
        ),
        (
            "Sempre il libro lì: fa parte dell'arredamento?",
            "Sempre il libro lì, fa parte dell'arredamento?",
            "Sempre il libro lì, fa parte dell'arredamento?",
        ),
        (
            "Banana, mela… manca solo l'arancia!",
            "Banana, mela, manca solo l'arancia!",
            "Banana, mela, manca solo l'arancia!",
        ),
        ("Mi chiamo ARI: so riconoscere oggetti.", "Mi chiamo ARI, so riconoscere oggetti", None),
        # Marks that are never spoken.
        ("Wait... davvero?!", "Wait, davvero?", None),
        ("Wow!!", "Wow!", None),
        ("Ma va?!?", "Ma va?", None),
        ('Dico «ciao» e "basta"; poi (forse) *rido*.', "Dico ciao e basta, poi forse rido", None),
        ("Due × tre.", "Due per tre", None),
        ("Che bello 😀👍🏽!", "Che bello!", None),
        ("Grazie ❤️ a tutti.", "Grazie a tutti", None),
        # Italian decimals and elisions stay as they are.
        ("Mi bastano 4,1 millisecondi.", "Mi bastano 4,1 millisecondi", None),
        ("C'è un'auto qui.", "C'è un'auto qui", None),
        ("  Ciao   a   tutti .  ", "Ciao a tutti", "Ciao a tutti."),
    ],
)
def test_normalize_real_phrases(text: str, xtts: str, other: str | None) -> None:
    assert normalize(text, "xtts") == xtts
    if other is not None:
        assert normalize(text, "kokoro") == other


def test_normalize_is_idempotent() -> None:
    for text in ("Sembra un'auto… dico bene?", "Ciao a tutti! Siete in 3.", "«Ehi» (ciao)!!"):
        once = normalize(text, "xtts")
        assert normalize(once, "xtts") == once


def test_sentences_keep_their_marks_and_numbers() -> None:
    assert sentences("Psst! Sì, dico proprio a te. Vieni!") == [
        "Psst!",
        "Sì, dico proprio a te.",
        "Vieni!",
    ]
    assert sentences("Già 10.000 persone. Wow") == ["Già 10.000 persone.", "Wow"]
    assert sentences("Ciao") == ["Ciao"]
