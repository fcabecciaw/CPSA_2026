"""Animated schematic demonstrations of the recorder prompts.

``draw_demo(panel, prompt_id, t)`` draws a looping figure (head, shoulders and
two hands) acting out the prompt, as the participant sees it in the mirrored
preview: the participant's right hand is on the right of the panel.
``STEPS`` holds short step-by-step instructions per prompt. ASCII only:
OpenCV's Hershey fonts cannot draw accented letters.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

INK = (235, 235, 235)
HAND = (60, 180, 255)  # amber (BGR)
TARGET = (90, 90, 255)
DIM = (120, 120, 120)

STEPS: dict[str, tuple[str, ...]] = {
    "neutral_hands_lap": ("Siediti comodo", "mani ferme sulle gambe", "guarda la camera"),
    "hands_open_front": ("Alza le due mani", "palmi verso la camera", "dita ben aperte"),
    "hands_move_free": ("Muovi le mani piano", "in ogni direzione", "dita aperte"),
    "pinch_right": ("Mano DESTRA davanti", "unisci pollice e indice", "poi riapri. Ripeti"),
    "pinch_left": ("Mano SINISTRA davanti", "unisci pollice e indice", "poi riapri. Ripeti"),
    "pinch_drag": ("Pizzica (pollice+indice)", "tieni chiuso e sposta", "la mano, poi rilascia"),
    "fist_open": ("Due mani davanti", "chiudi a pugno", "poi apri tutto. Ripeti"),
    "two_hands_zoom": ("Pizzica con DUE mani", "allontana le mani...", "...poi riavvicinale"),
    "palms_explode": ("Palmi aperti al centro", "spingili veloce in fuori", "torna piano. Ripeti"),
    "fists_together": ("Pugni lontani ai lati", "avvicinali al centro", "fino a toccarsi"),
    "swipe": ("Mano aperta", "scorri veloce a destra", "e poi a sinistra"),
    "point": ("Dito indice teso", "indica punti diversi", "dello schermo"),
    "rotate_wrists": ("Mani davanti", "ruota i polsi", "palmo, dorso, palmo"),
    "hand_mouth": ("Piano e delicato:", "mano verso la bocca", "poi allontana"),
    "hand_eye": ("Piano e delicato:", "dita vicino all'occhio", "poi allontana"),
    "hand_cheek": ("Piano e delicato:", "mano sulla guancia", "poi allontana"),
    "hand_forehead": ("Piano e delicato:", "mano sulla fronte", "poi allontana"),
    "hand_ear": ("Piano:", "tocca l'orecchio", "poi allontana"),
    "scratch_face": ("Gratta piano", "il naso o la guancia", "con le dita"),
    "hands_over_face": ("Copri il viso", "con le due mani", "poi scopri"),
    "chin_rest": ("Appoggia il mento", "sulla mano", "resta fermo"),
    "rocking": ("Dondola piano", "avanti e indietro", "con il busto"),
    "hand_flapping": ("Mani vicino alle spalle", "agitale avanti e indietro", "velocemente"),
    "drink_phone": ("Fai finta di bere", "poi di parlare", "al telefono"),
    "lean_back": ("Appoggiati indietro", "sullo schienale", "mani visibili"),
    "partly_out": ("Muovi le mani", "verso i bordi", "anche un po' fuori"),
    "turn_head": ("Gira la testa", "a destra e a sinistra", "mani davanti"),
}


def _tri(phase: float) -> float:
    """Triangle wave 0..1..0 over one period."""
    return 1.0 - abs(2.0 * (phase % 1.0) - 1.0)


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _hand(
    img: np.ndarray,
    center: tuple[float, float],
    size: float,
    *,
    open_amount: float = 1.0,
    pinch: float = 0.0,
    point: bool = False,
    right: bool = True,
    width: float = 1.0,
    color: tuple[int, int, int] = HAND,
) -> None:
    """Draw a schematic hand: palm plus five finger rays.

    ``open_amount`` 0 is a fist and 1 an open hand. ``pinch`` 0..1 brings the
    thumb and index tips together. ``point`` extends only the index finger.
    ``width`` scales x (wrist rotation).
    """
    cx, cy = center
    sign = 1.0 if right else -1.0
    palm_w = size * 0.55 * max(width, 0.15)
    cv2.ellipse(
        img, (int(cx), int(cy)), (int(palm_w), int(size * 0.6)), 0, 0, 360, color, 2, cv2.LINE_AA
    )
    finger_angles = (-24, -8, 8, 24)  # index, middle, ring, little (degrees from vertical)
    tips = []
    for i, angle in enumerate(finger_angles):
        rad = math.radians(angle * sign)
        amount = (1.0 if i == 0 else 0.15) if point else open_amount
        length = size * (0.35 + 0.85 * amount)
        bx = cx + sign * palm_w * 0.8 * (i - 1.5) / 1.5
        by = cy - size * 0.5
        tx = bx + math.sin(rad) * length * max(width, 0.15)
        ty = by - math.cos(rad) * length
        if i == 0 and pinch > 0:
            thumb_tip = (cx - sign * palm_w * 1.2, cy - size * 0.35)
            tx = tx + (thumb_tip[0] - tx) * pinch * 0.8
            ty = ty + (thumb_tip[1] - ty) * pinch * 0.8
        tips.append((tx, ty))
        cv2.line(img, (int(bx), int(by)), (int(tx), int(ty)), color, 3, cv2.LINE_AA)
    # Thumb, from the side of the palm.
    tbx, tby = cx - sign * palm_w * 0.9, cy
    thumb_len = size * (0.3 + 0.6 * (open_amount if not point else 0.2))
    ttx = tbx - sign * thumb_len * 0.8 * max(width, 0.15)
    tty = tby - thumb_len * 0.7
    if pinch > 0 and tips:
        ttx = ttx + (tips[0][0] - ttx) * pinch * 0.8
        tty = tty + (tips[0][1] - tty) * pinch * 0.8
    cv2.line(img, (int(tbx), int(tby)), (int(ttx), int(tty)), color, 3, cv2.LINE_AA)
    if pinch > 0.85:
        cv2.circle(img, (int(ttx), int(tty)), 6, (255, 255, 255), 2, cv2.LINE_AA)


def _figure(img: np.ndarray, head: tuple[float, float], scale: float = 1.0) -> dict[str, tuple]:
    """Head, face features and shoulders. Returns the face landmark points."""
    hx, hy = head
    r = 46 * scale
    cv2.circle(img, (int(hx), int(hy)), int(r), INK, 2, cv2.LINE_AA)
    eye_l, eye_r = (hx - r * 0.38, hy - r * 0.15), (hx + r * 0.38, hy - r * 0.15)
    for ex, ey in (eye_l, eye_r):
        cv2.circle(img, (int(ex), int(ey)), int(5 * scale), INK, -1, cv2.LINE_AA)
    mouth = (hx, hy + r * 0.45)
    cv2.ellipse(
        img, (int(mouth[0]), int(mouth[1])), (int(r * 0.3), int(r * 0.08)), 0, 0, 360, INK, 2
    )
    nose = (hx, hy + r * 0.12)
    sy = hy + r * 1.6
    cv2.line(img, (int(hx - r * 2.2), int(sy)), (int(hx + r * 2.2), int(sy)), DIM, 2, cv2.LINE_AA)
    cv2.line(img, (int(hx), int(hy + r)), (int(hx), int(sy)), DIM, 2, cv2.LINE_AA)
    return {
        "eye": eye_r,
        "mouth": mouth,
        "nose": nose,
        "cheek": (hx + r * 0.62, hy + r * 0.25),
        "forehead": (hx, hy - r * 0.7),
        "ear": (hx + r * 1.0, hy),
        "chin": (hx, hy + r * 0.95),
        "r": r,
    }


def _arrow(img: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> None:
    cv2.arrowedLine(
        img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), DIM, 2, cv2.LINE_AA, tipLength=0.25
    )


def _approach(
    img: np.ndarray, face: dict[str, tuple], target: str, t: float, period: float = 3.0
) -> None:
    """Right hand travels from rest to a face target and back."""
    goal = face[target]
    rest = (goal[0] + 90, goal[1] + 150)
    k = _smooth(_tri(t / period) * 1.4)
    pos = (rest[0] + (goal[0] - rest[0]) * k, rest[1] + (goal[1] - rest[1]) * k)
    cv2.circle(img, (int(goal[0]), int(goal[1])), 9, TARGET, 2, cv2.LINE_AA)
    _hand(img, pos, 34, open_amount=0.6)


def draw_demo(panel: np.ndarray, prompt_id: str, t: float) -> None:
    """Draw the looping demonstration for ``prompt_id`` at time ``t`` seconds."""
    panel[:] = (18, 18, 22)
    h, w = panel.shape[:2]
    cx = w / 2
    head = (cx, h * 0.30)
    ph = t % 4.0 / 4.0
    hand_y = h * 0.62
    left_x, right_x = cx - w * 0.25, cx + w * 0.25

    if prompt_id == "rocking":
        s = 1.0 + 0.12 * math.sin(t * 2.0)
        face = _figure(panel, (cx, h * 0.30 + 10 * math.sin(t * 2.0)), s)
        _arrow(panel, (cx + 120, h * 0.2), (cx + 120, h * 0.42))
        return
    if prompt_id == "turn_head":
        face = _figure(panel, (cx + 40 * math.sin(t * 1.6), head[1]))
    elif prompt_id == "lean_back":
        face = _figure(panel, (cx, head[1] - 10), 0.8)
    else:
        face = _figure(panel, head)

    if prompt_id == "neutral_hands_lap":
        _hand(panel, (left_x, h * 0.88), 30, right=False, open_amount=0.7)
        _hand(panel, (right_x, h * 0.88), 30, open_amount=0.7)
    elif prompt_id in ("hands_open_front", "lean_back", "turn_head"):
        dy = 6 * math.sin(t * 2)
        _hand(panel, (left_x, hand_y + dy), 36, right=False)
        _hand(panel, (right_x, hand_y - dy), 36)
    elif prompt_id == "hands_move_free":
        _hand(panel, (left_x + 40 * math.sin(t), hand_y + 50 * math.sin(t * 1.3)), 34, right=False)
        _hand(panel, (right_x + 40 * math.cos(t * 1.1), hand_y + 50 * math.cos(t)), 34)
    elif prompt_id in ("pinch_right", "pinch_left"):
        right = prompt_id == "pinch_right"
        x = right_x if right else left_x
        _hand(panel, (x, hand_y), 42, right=right, pinch=_smooth(_tri(t / 1.6) * 1.5))
        _hand(panel, (left_x if right else right_x, h * 0.88), 26, right=not right, color=DIM)
    elif prompt_id == "pinch_drag":
        k = _tri(t / 4.0)
        pinch = 1.0 if 0.1 < k < 0.9 else _smooth(k * 10 if k <= 0.1 else (1 - k) * 10)
        _hand(panel, (right_x - 120 * k, hand_y - 60 * math.sin(math.pi * k)), 40, pinch=pinch)
        _arrow(panel, (right_x, hand_y + 70), (right_x - 120, hand_y + 70))
    elif prompt_id == "fist_open":
        amount = _smooth(_tri(t / 1.8) * 1.3)
        _hand(panel, (left_x, hand_y), 38, right=False, open_amount=amount)
        _hand(panel, (right_x, hand_y), 38, open_amount=amount)
    elif prompt_id == "two_hands_zoom":
        spread = 40 + 90 * _smooth(_tri(t / 3.0) * 1.2)
        _hand(panel, (cx - spread, hand_y), 34, right=False, pinch=1.0)
        _hand(panel, (cx + spread, hand_y), 34, pinch=1.0)
        _arrow(panel, (cx - 30, hand_y + 70), (cx - 130, hand_y + 70))
        _arrow(panel, (cx + 30, hand_y + 70), (cx + 130, hand_y + 70))
    elif prompt_id == "palms_explode":
        k = ph
        out = _smooth(min(k / 0.25, 1.0)) if k < 0.5 else 1.0 - _smooth((k - 0.5) / 0.5)
        spread = 35 + 110 * out
        _hand(panel, (cx - spread, hand_y), 38, right=False)
        _hand(panel, (cx + spread, hand_y), 38)
        if 0.05 < k < 0.3:
            for r in (40, 70, 100):
                cv2.circle(panel, (int(cx), int(hand_y)), int(r * (k / 0.3)), DIM, 1, cv2.LINE_AA)
    elif prompt_id == "fists_together":
        spread = 30 + 120 * (1.0 - _smooth(_tri(t / 3.0) * 1.2))
        _hand(panel, (cx - spread, hand_y), 34, right=False, open_amount=0.0)
        _hand(panel, (cx + spread, hand_y), 34, open_amount=0.0)
        _arrow(panel, (cx - 150, hand_y + 70), (cx - 50, hand_y + 70))
        _arrow(panel, (cx + 150, hand_y + 70), (cx + 50, hand_y + 70))
    elif prompt_id == "swipe":
        x = cx + 120 * math.sin(t * 2.4)
        _hand(panel, (x, hand_y), 38)
        _arrow(panel, (cx - 120, hand_y + 70), (cx + 120, hand_y + 70))
    elif prompt_id == "point":
        targets = [(cx - 110, h * 0.12), (cx + 110, h * 0.12), (cx + 120, h * 0.5)]
        goal = targets[int(t / 1.5) % len(targets)]
        cv2.circle(panel, (int(goal[0]), int(goal[1])), 10, TARGET, 2, cv2.LINE_AA)
        x = right_x + (goal[0] - right_x) * 0.25
        _hand(panel, (x, hand_y), 40, point=True)
    elif prompt_id == "rotate_wrists":
        width = math.cos(t * 2.0)
        color = HAND if width >= 0 else DIM
        _hand(panel, (left_x, hand_y), 38, right=False, width=abs(width), color=color)
        _hand(panel, (right_x, hand_y), 38, width=abs(width), color=color)
    elif prompt_id in ("hand_mouth", "hand_eye", "hand_cheek", "hand_forehead", "hand_ear"):
        _approach(panel, face, prompt_id.split("_", 1)[1], t)
    elif prompt_id == "scratch_face":
        nose = face["nose"]
        _hand(panel, (nose[0] + 20 + 8 * math.sin(t * 9), nose[1] + 30), 30, open_amount=0.3)
    elif prompt_id == "hands_over_face":
        k = _smooth(_tri(t / 3.0) * 1.3)
        _hand(panel, (cx - 30 - 90 * (1 - k), head[1] + 200 * (1 - k)), 40, right=False)
        _hand(panel, (cx + 30 + 90 * (1 - k), head[1] + 200 * (1 - k)), 40)
    elif prompt_id == "chin_rest":
        chin = face["chin"]
        _hand(panel, (chin[0], chin[1] + 30), 32, open_amount=0.2)
    elif prompt_id == "hand_flapping":
        dy = 12 * math.sin(t * 10)
        _hand(panel, (cx - face["r"] * 1.9, h * 0.45 + dy), 30, right=False)
        _hand(panel, (cx + face["r"] * 1.9, h * 0.45 - dy), 30)
    elif prompt_id == "drink_phone":
        k = _smooth(_tri(t / 3.0) * 1.3)
        goal = face["mouth"] if int(t / 6.0) % 2 == 0 else face["ear"]
        pos = (right_x + (goal[0] + 20 - right_x) * k, hand_y + (goal[1] + 20 - hand_y) * k)
        cv2.rectangle(
            panel, (int(pos[0] - 10), int(pos[1] - 40)), (int(pos[0] + 10), int(pos[1])), DIM, 2
        )
        _hand(panel, pos, 30, open_amount=0.3)
    elif prompt_id == "partly_out":
        x = cx + (w * 0.55) * math.sin(t * 1.2)
        _hand(panel, (x, hand_y), 40)
    else:
        _hand(panel, (left_x, hand_y), 34, right=False)
        _hand(panel, (right_x, hand_y), 34)


def draw_steps(panel: np.ndarray, prompt_id: str, top: int) -> None:
    """Write the step-by-step lines under the animation."""
    for i, line in enumerate(STEPS.get(prompt_id, ())):
        cv2.putText(
            panel,
            line,
            (14, top + 26 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
