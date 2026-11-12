#!/usr/bin/env python3
"""Re-stamp the DPU fingerprint of an already-compiled ``.xmodel`` (host, no xir).

``tools/vitis/compile.sh`` stamps the KV260 fingerprint ``0x101000016010407``
into the artifact it compiles, because Vitis AI 3.5's target alias
``DPUCZDX8G_ISA1_B4096`` maps to ``0x101000056010407`` and the board's DPU
firmware rejects that value.  The model-zoo YOLOv5-nano deployment
(``yolov5_nano_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz``) was compiled for the very
same target by Vitis AI 3.0 and records the 3.x value, so it is rejected for the
same reason even though its instructions target ``DPUCZDX8G_ISA1_B4096``.

The only difference is the recorded attribute: an ``xmodel`` is a protobuf graph
whose subgraph attribute ``dpu_fingerprint`` is a protobuf varint.  This tool
rewrites that varint in place, without touching the graph, the weights or the
DPU instructions, and refuses to write anything unless the source varint occurs
exactly once in the file and the target varint has the same length.  Verify the
result with ``inspect_xmodel.py`` (it reads the attribute back through ``xir``).

Example::

    uv run --no-sync python tools/vitis/restamp_xmodel.py \\
        build/models/zoo/yolov5_nano/yolov5_nano_pt/yolov5_nano_pt.xmodel \\
        build/vitis/yolov5_nano/compiled/yolov5_nano.xmodel
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: Fingerprint Vitis AI 3.0/3.5 records for DPUCZDX8G_ISA1_B4096.
DEFAULT_SOURCE_FINGERPRINT = 0x101000056010407

#: Fingerprint the KV260 B4096 firmware reports (``xdputil query``).
DEFAULT_TARGET_FINGERPRINT = 0x101000016010407


def varint(value: int) -> bytes:
    """Encode ``value`` as a protobuf base-128 varint."""

    if value < 0:
        raise ValueError(f"a fingerprint is unsigned, got {value}")
    encoded = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            encoded.append(chunk | 0x80)
        else:
            encoded.append(chunk)
            return bytes(encoded)


def restamp(
    source: Path, target: Path, source_fingerprint: int, fingerprint: int
) -> dict[str, int]:
    """Rewrite one fingerprint varint of ``source`` into ``target``.

    Returns:
        ``{"occurrences": n, "offset": byte offset, "bytes": varint length}``.

    Raises:
        ValueError: If the fingerprints differ, the varint lengths differ or the
            source varint is not in the file exactly once.
    """

    if source_fingerprint == fingerprint:
        raise ValueError("source and target fingerprints are identical")
    old = varint(source_fingerprint)
    new = varint(fingerprint)
    if len(old) != len(new):
        raise ValueError(
            f"the varint encodings differ in length ({len(old)} vs {len(new)} bytes); "
            "this tool only rewrites a same-length attribute"
        )
    data = source.read_bytes()
    occurrences = data.count(old)
    if occurrences != 1:
        raise ValueError(
            f"expected the fingerprint {hex(source_fingerprint)} exactly once in {source}, "
            f"found it {occurrences} times"
        )
    offset = data.index(old)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data[:offset] + new + data[offset + len(old) :])
    return {"occurrences": occurrences, "offset": offset, "bytes": len(new)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="compiled .xmodel to re-stamp")
    parser.add_argument("target", type=Path, help="where to write the re-stamped .xmodel")
    parser.add_argument(
        "--source-fingerprint",
        type=lambda text: int(text, 0),
        default=DEFAULT_SOURCE_FINGERPRINT,
        help="fingerprint recorded in the source (default: the 3.x B4096 value)",
    )
    parser.add_argument(
        "--fingerprint",
        type=lambda text: int(text, 0),
        default=DEFAULT_TARGET_FINGERPRINT,
        help="fingerprint to record (default: the KV260 B4096 firmware value)",
    )
    args = parser.parse_args(argv)
    if not args.source.is_file():
        print(f"restamp_xmodel.py: {args.source} not found")
        return 1
    try:
        result = restamp(args.source, args.target, args.source_fingerprint, args.fingerprint)
    except ValueError as exc:
        print(f"restamp_xmodel.py: {exc}")
        return 1
    print(
        f"restamp_xmodel.py: {hex(args.source_fingerprint)} -> {hex(args.fingerprint)} "
        f"at offset {result['offset']} ({result['bytes']} bytes) in {args.target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
