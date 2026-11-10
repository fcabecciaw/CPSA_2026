"""Generate a printable ChArUco calibration board as an exact-millimetre vector PDF.

The board layout comes from OpenCV itself (``cv2.aruco.CharucoBoard``, non-legacy
pattern), so the printed target matches what ``cv2.aruco.CharucoDetector`` expects.
OpenCV renders the board as a raster on an integer cell grid; every cell is then
emitted as a vector rectangle in the PDF, so edges are exact at any printer
resolution. The PDF is rasterised back and detected to check that every
ChArUco corner is found.

Usage::

    uv run --no-project --with opencv-contrib-python-headless --with reportlab \
        --with pymupdf python tools/calibration/charuco_board.py --preset a3 --out board.pdf
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# Square and marker sizes are chosen so the marker is exactly 0.7 of the square:
# with 20 raster cells per square the marker starts 3 cells in and each of its
# 7 modules (5 data bits + 1 border each side) is exactly 2 cells wide.
CELLS_PER_SQUARE = 20
MARKER_RATIO = 0.7


@dataclass(frozen=True)
class Preset:
    page_w_mm: float
    page_h_mm: float
    squares_x: int  # columns
    squares_y: int  # rows
    square_mm: float


PRESETS = {
    "a3": Preset(420.0, 297.0, 11, 8, 30.0),
    "a4": Preset(297.0, 210.0, 9, 7, 25.0),
}
DICTIONARY_NAME = "DICT_5X5_100"


def make_board(p: Preset) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))
    board = cv2.aruco.CharucoBoard(
        (p.squares_x, p.squares_y),
        p.square_mm / 1000.0,
        p.square_mm * MARKER_RATIO / 1000.0,
        dictionary,
    )
    board.setLegacyPattern(False)
    return board


def cell_grid(board: cv2.aruco.CharucoBoard, p: Preset, px_per_cell: int = 8) -> np.ndarray:
    """Return a boolean (rows, cols) grid of cells, True where black."""
    px_per_square = CELLS_PER_SQUARE * px_per_cell
    size = (p.squares_x * px_per_square, p.squares_y * px_per_square)
    img = board.generateImage(size, marginSize=0, borderBits=1)
    centres = img[px_per_cell // 2 :: px_per_cell, px_per_cell // 2 :: px_per_cell]
    # Every cell must be uniform, otherwise the cell grid does not match the layout.
    blocks = img.reshape(size[1] // px_per_cell, px_per_cell, size[0] // px_per_cell, px_per_cell)
    if not (blocks.min(axis=(1, 3)) == blocks.max(axis=(1, 3))).all():
        raise RuntimeError("board raster is not aligned to the cell grid")
    return centres < 128


def write_pdf(grid: np.ndarray, p: Preset, out: Path) -> None:
    cell_mm = p.square_mm / CELLS_PER_SQUARE
    board_w = p.squares_x * p.square_mm
    board_h = p.squares_y * p.square_mm
    x0 = (p.page_w_mm - board_w) / 2
    y0 = (p.page_h_mm - board_h) / 2  # bottom edge of the board, PDF origin is bottom-left
    c = canvas.Canvas(str(out), pagesize=(p.page_w_mm * mm, p.page_h_mm * mm))
    c.setTitle(f"ChArUco {p.squares_x}x{p.squares_y} {p.square_mm:g}mm {DICTIONARY_NAME}")
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0, 0, 0)
    # All black cells go into one path filled once, so adjacent cells merge without
    # the anti-aliasing seams that separate rectangles show in viewers and RIPs.
    path = c.beginPath()
    rows, cols = grid.shape
    for r in range(rows):
        cc = 0
        while cc < cols:  # merge horizontal runs of black cells into one rectangle
            if not grid[r, cc]:
                cc += 1
                continue
            start = cc
            while cc < cols and grid[r, cc]:
                cc += 1
            x = x0 + start * cell_mm
            y = y0 + board_h - (r + 1) * cell_mm
            path.rect(x * mm, y * mm, (cc - start) * cell_mm * mm, cell_mm * mm)
    c.drawPath(path, stroke=0, fill=1)

    # Scale check bar (100 mm) and parameters in the bottom margin.
    margin_y = y0 / 2
    c.setLineWidth(0.3)
    c.line(x0 * mm, margin_y * mm, (x0 + 100) * mm, margin_y * mm)
    for t in range(0, 101, 10):
        h = 3 if t % 50 == 0 else 1.5
        c.line((x0 + t) * mm, margin_y * mm, (x0 + t) * mm, (margin_y + h) * mm)
    c.setFont("Helvetica", 7)
    c.drawString((x0 + 103) * mm, (margin_y - 0.8) * mm, "100 mm: measure after printing")
    c.drawRightString(
        (x0 + board_w) * mm,
        (margin_y - 0.8) * mm,
        f"ChArUco {p.squares_x}x{p.squares_y} squares, square {p.square_mm:g} mm, "
        f"marker {p.square_mm * MARKER_RATIO:g} mm, {DICTIONARY_NAME}, ids from 0, "
        f"OpenCV non-legacy. Print at 100% (no fit to page).",
    )
    c.showPage()
    c.save()


def verify_pdf(board: cv2.aruco.CharucoBoard, p: Preset, out: Path, dpi: int = 150) -> int:
    """Rasterise the PDF and return the number of ChArUco corners detected."""
    import pymupdf

    page = pymupdf.open(str(out))[0]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
    if ids is None:
        return 0
    # Check the measured square size on the rendered page.
    px_per_mm = dpi / 25.4
    pts = corners.reshape(-1, 2)
    by_id = dict(zip(ids.flatten().tolist(), pts, strict=True))
    if 0 in by_id and 1 in by_id:
        d_mm = float(np.linalg.norm(by_id[1] - by_id[0])) / px_per_mm
        print(f"rendered square size: {d_mm:.2f} mm (expected {p.square_mm:g})")
    return len(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="a3")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = PRESETS[args.preset]
    board = make_board(p)
    write_pdf(cell_grid(board, p), p, args.out)
    expected = (p.squares_x - 1) * (p.squares_y - 1)
    found = verify_pdf(board, p, args.out)
    print(f"{args.out}: {found}/{expected} ChArUco corners detected")
    return 0 if found == expected else 1


if __name__ == "__main__":
    sys.exit(main())
