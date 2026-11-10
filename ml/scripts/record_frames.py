#!/usr/bin/env python3
"""Guided frame recorder for the CPSA booth datasets (body and hand models).

Shows a mirrored live preview of the RGB camera with Italian prompts. Each
prompt has a countdown, then frames are saved at ``--save-fps`` (not every
camera frame, to avoid near-duplicates) as JPEG files, with one metadata line
per frame in ``frames.jsonl``. The frames are later auto-labeled by the
teacher model and used for validation and fine-tuning.

Run with the ml environment (OpenCV with GUI)::

    ml/.venv/bin/python ml/scripts/record_frames.py --participant P01 --out data/booth

Keys: SPACE pause/resume, N skip prompt, R repeat the last prompt,
F free recording (toggle), Q quit. ``--rehearse`` shows the prompts and
animations without saving anything.

The recordings contain faces: record only people who agreed to it, keep the
data local (``data/`` is git-ignored) and do not publish it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from gesture_demo import draw_demo, draw_steps


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    seconds: float
    group: str


# Order: warm-up, model-lab gestures, medical-like hand-to-face (gentle), negatives.
DEFAULT_PROMPTS: tuple[Prompt, ...] = (
    Prompt("neutral_hands_lap", "Mani appoggiate sulle gambe, guarda la camera", 6, "neutral"),
    Prompt("hands_open_front", "Mostra entrambe le mani aperte verso la camera", 8, "lab"),
    Prompt("hands_move_free", "Muovi le mani lentamente davanti a te, dita aperte", 10, "lab"),
    Prompt("pinch_right", "Mano destra: pizzica (pollice e indice) e rilascia", 8, "lab"),
    Prompt("pinch_left", "Mano sinistra: pizzica e rilascia, piu volte", 8, "lab"),
    Prompt("pinch_drag", "Pizzica e trascina come per ruotare un oggetto", 10, "lab"),
    Prompt("fist_open", "Chiudi e apri le mani a pugno, piu volte", 8, "lab"),
    Prompt("two_hands_zoom", "Due mani che pizzicano: allontanale e avvicinale", 10, "lab"),
    Prompt("palms_explode", "Palmi aperti al centro, poi spingili in fuori veloce", 10, "lab"),
    Prompt("fists_together", "Pugni lontani, poi avvicinali al centro", 10, "lab"),
    Prompt("swipe", "Mano aperta: scorri a destra e a sinistra", 10, "lab"),
    Prompt("point", "Indica con l'indice punti diversi dello schermo", 8, "lab"),
    Prompt("rotate_wrists", "Ruota i polsi, mostra il dorso e il palmo", 8, "lab"),
    Prompt("hand_mouth", "Lentamente: mano verso la bocca e via (delicato)", 10, "medical"),
    Prompt("hand_eye", "Lentamente: dita vicino all'occhio e via (delicato)", 10, "medical"),
    Prompt("hand_cheek", "Mano sulla guancia, poi via (delicato)", 10, "medical"),
    Prompt("hand_forehead", "Mano sulla fronte, poi via (delicato)", 8, "medical"),
    Prompt("hand_ear", "Tocca l'orecchio, poi via", 8, "medical"),
    Prompt("scratch_face", "Gratta il viso o il naso", 8, "medical"),
    Prompt("hands_over_face", "Copri il viso con le mani, poi scopri", 8, "medical"),
    Prompt("chin_rest", "Appoggia il mento sulla mano", 8, "medical"),
    Prompt("rocking", "Dondola lentamente avanti e indietro", 10, "medical"),
    Prompt("hand_flapping", "Agita le mani vicino alle spalle", 8, "medical"),
    Prompt("drink_phone", "Fai finta di bere o di usare il telefono", 10, "negative"),
    Prompt("lean_back", "Appoggiati indietro, mani visibili", 8, "negative"),
    Prompt("partly_out", "Mani parzialmente fuori dall'inquadratura", 8, "negative"),
    Prompt("turn_head", "Gira la testa a destra e a sinistra, mani davanti", 8, "neutral"),
)

COUNTDOWN_S = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--participant", required=True, help="pseudonymous id, e.g. P01")
    parser.add_argument("--out", type=Path, default=Path("data/booth"), help="output root")
    parser.add_argument("--device", default="/dev/video4", help="V4L2 RGB node (F200: video4)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="camera frame rate")
    parser.add_argument("--save-fps", type=float, default=5.0, help="frames saved per second")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--scale", type=float, default=1.5, help="preview window scale")
    parser.add_argument(
        "--exposure",
        type=int,
        default=300,
        help="manual exposure_time_absolute (100 us units; board config uses 300; 0 = auto)",
    )
    parser.add_argument("--gain", type=int, default=None, help="sensor gain (64..2540)")
    parser.add_argument("--only", nargs="*", default=None, help="run only these prompt ids")
    parser.add_argument("--list", action="store_true", help="print the prompts and exit")
    parser.add_argument(
        "--rehearse", action="store_true", help="show prompts and animations, save nothing"
    )
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    # Same camera settings as the board: manual exposure (less motion blur, stable
    # 30 fps) and 50 Hz mains filtering against flicker banding.
    controls = ["power_line_frequency=1"]
    if args.exposure > 0:
        controls += ["auto_exposure=1", f"exposure_time_absolute={args.exposure}"]
    if args.gain is not None:
        controls.append(f"gain={args.gain}")
    for control in controls:
        subprocess.run(["v4l2-ctl", "-d", args.device, "-c", control], check=False)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit(f"camera {args.device} delivered no frame")
    h, w = frame.shape[:2]
    if (w, h) != (args.width, args.height):
        raise SystemExit(f"camera gave {w}x{h}, expected {args.width}x{args.height}")
    return cap


def draw_overlay(
    view: np.ndarray, title: str, sub: str, recording: bool, saved: int, paused: bool
) -> None:
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.putText(view, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(view, sub, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if recording:
        cv2.circle(view, (w - 24, 24), 10, (0, 0, 255), -1)
    status = "PAUSA" if paused else f"salvati: {saved}"
    cv2.putText(view, status, (w - 170, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)


def main() -> int:
    args = parse_args()
    prompts = [p for p in DEFAULT_PROMPTS if args.only is None or p.id in args.only]
    if args.list:
        for p in DEFAULT_PROMPTS:
            print(f"{p.id:20s} {p.group:9s} {p.seconds:4.0f}s  {p.text}")
        return 0
    if not prompts:
        raise SystemExit("no prompts selected")

    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out / f"{args.participant}_{session}"
    if args.rehearse:
        root = Path("/tmp") / f"cpsa_rehearse_{session}"
    root.mkdir(parents=True, exist_ok=False)
    meta_path = root / "frames.jsonl"
    (root / "session.json").write_text(
        json.dumps(
            {
                "participant": args.participant,
                "session": session,
                "device": args.device,
                "size": [args.width, args.height],
                "camera_fps": args.fps,
                "save_fps": args.save_fps,
                "prompts": [p.__dict__ for p in prompts],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    cap = open_camera(args)
    window = "CPSA recorder"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, int((args.width + 380) * args.scale), int(args.height * args.scale))
    params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    saved = 0
    free_index = 0
    index = 0
    paused = False
    free = False
    phase_start = time.monotonic()
    last_save = 0.0
    period = 1.0 / args.save_fps

    with meta_path.open("a", encoding="utf-8") as meta:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("camera read failed; stopping")
                break
            now = time.monotonic()
            if index >= len(prompts) and not free:
                break
            prompt = prompts[min(index, len(prompts) - 1)]
            elapsed = now - phase_start
            recording = False
            if free:
                title, sub = "REGISTRAZIONE LIBERA", "F per fermare"
                prompt_id, group = f"free_{free_index:02d}", "free"
                recording = not paused
            elif paused:
                title, sub = prompt.text, "In pausa: SPAZIO per riprendere"
                prompt_id, group = prompt.id, prompt.group
                phase_start = now - min(elapsed, COUNTDOWN_S)
            elif elapsed < COUNTDOWN_S:
                remaining = COUNTDOWN_S - elapsed
                title = prompt.text
                sub = f"[{index + 1}/{len(prompts)}] si parte tra {remaining:.0f}..."
                prompt_id, group = prompt.id, prompt.group
            elif elapsed < COUNTDOWN_S + prompt.seconds:
                remaining = COUNTDOWN_S + prompt.seconds - elapsed
                title, sub = prompt.text, f"[{index + 1}/{len(prompts)}] REC {remaining:.0f}s"
                prompt_id, group = prompt.id, prompt.group
                recording = True
            else:
                index += 1
                phase_start = now
                continue

            if recording and not args.rehearse and now - last_save >= period:
                last_save = now
                folder = root / prompt_id
                folder.mkdir(exist_ok=True)
                name = f"{prompt_id}_{saved:05d}.jpg"
                cv2.imwrite(str(folder / name), frame, params)
                meta.write(
                    json.dumps(
                        {
                            "file": f"{prompt_id}/{name}",
                            "t": now,
                            "wall": datetime.now(timezone.utc).isoformat(),
                            "prompt": prompt_id,
                            "group": group,
                            "participant": args.participant,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                saved += 1

            view = cv2.flip(frame, 1)  # mirror for the participant; saved frames are not mirrored
            draw_overlay(view, title, sub, recording, saved, paused)
            panel = np.zeros((view.shape[0], 380, 3), np.uint8)
            demo_id = prompt_id if not free else ""
            draw_demo(panel[: view.shape[0] - 110], demo_id, now - phase_start)
            draw_steps(panel, demo_id, view.shape[0] - 85)
            cv2.imshow(window, np.hstack([view, panel]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("n") and not free:
                index += 1
                phase_start = time.monotonic()
            elif key == ord("r") and not free:
                index = max(0, index - 1) if elapsed < COUNTDOWN_S else index
                phase_start = time.monotonic()
            elif key == ord("f"):
                free = not free
                free_index += 0 if free else 1
                phase_start = time.monotonic()

    cap.release()
    cv2.destroyAllWindows()
    print(f"saved {saved} frames to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
