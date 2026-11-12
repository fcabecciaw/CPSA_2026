"""Command line of the local TTS: serve, pregenerate, samples, bench.

``cpsa-tts --help`` lists them. The dashboard only needs ``serve``; the other
commands exist so the user can listen to both engines before choosing one and so
the fixed dashboard phrases are already in the disk cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .cache import WavCache, write_wav_atomic
from .config import Settings, repo_root
from .defaults import DefaultStore, apply_persisted
from .engines import Engine, EngineError, EngineRegistry, Voice
from .phrases import PhrasesDocument, collect, write_phrases

log = logging.getLogger("cpsa_tts.cli")

#: Sentinels for the cache so ``bench`` measures a real synthesis every time.
BENCH_TEXTS = (
    "Ciao! Sono il robot della mostra e vi guardo sessanta volte al secondo.",
    "Il polpo ha gran parte dei suoi neuroni nei tentacoli: ogni braccio pensa da solo.",
    "Mi chiamo ARI: so riconoscere oggetti, movimenti e sorrisi.",
)

SAMPLE_TEXTS = (
    ("saluto", "Ciao! Che piacere vederti."),
    (
        "fatto-demo",
        "Guardo il mondo sessanta volte al secondo, e ogni volta cerco ottanta "
        "tipi di oggetti diversi.",
    ),
    (
        "fatto-polpo",
        "Il polpo ha gran parte dei suoi neuroni nei tentacoli: ogni braccio, "
        "in un certo senso, pensa da solo.",
    ),
    ("oggetto", "Vedo il cellulare! Chissà quante notifiche."),
    ("congedo", "Ciao, a presto! Grazie della visita."),
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, (args.log_level or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # espeak/phonemizer emit a warning per line; the dashboard log stays readable.
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    settings = _settings(args)
    handlers = {
        "serve": _serve,
        "pregenerate": _pregenerate,
        "samples": _samples,
        "bench": _bench,
        "synthesize": _synthesize,
        "voices": _voices,
        "phrases": _phrases,
    }
    try:
        return handlers[args.command](args, settings)
    except EngineError as error:
        log.error("%s", error)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive convenience
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpsa-tts",
        description="Local Italian TTS service for the CPSA showcase (Kokoro, XTTS-v2, Edge).",
    )
    parser.add_argument("--version", action="version", version=f"cpsa-tts {__version__}")
    parser.add_argument("--cache-dir", type=Path, help="WAV cache (default ~/.cache/cpsa-tts)")
    parser.add_argument("--log-level", help="DEBUG, INFO, WARNING (default INFO)")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service on 127.0.0.1:8765")
    serve.add_argument("--host", help="bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port (default 8765)")
    serve.add_argument(
        "--warm",
        action="store_true",
        help="load the default engine before accepting requests",
    )

    pre = sub.add_parser("pregenerate", help="fill the cache with the dashboard's fixed phrases")
    pre.add_argument(
        "--engine", action="append", dest="engines", help="repeatable; default: the default engine"
    )
    pre.add_argument("--voice", help="one voice of that engine (default: its first voice)")
    pre.add_argument("--limit", type=int, help="max sentences per engine/voice")
    pre.add_argument("--dry-run", action="store_true", help="only export phrases.json")
    pre.add_argument(
        "--json", type=Path, help="path of phrases.json (default <repo>/build/phrases.json)"
    )
    pre.add_argument("--report", type=Path, help="path of the run report JSON")

    samples = sub.add_parser("samples", help="write comparison WAVs (5 sentences, every voice)")
    samples.add_argument(
        "--engine", action="append", dest="engines", help="repeatable; default: all"
    )
    samples.add_argument(
        "--voice", action="append", dest="voices", help="repeatable; default: every voice"
    )
    samples.add_argument(
        "--out", type=Path, help="output directory (default <repo>/build/tts-samples)"
    )
    samples.add_argument("--force", action="store_true", help="overwrite existing WAVs")

    bench = sub.add_parser("bench", help="measure cold and warm latency per engine")
    bench.add_argument("--engine", action="append", dest="engines", help="repeatable; default: all")
    bench.add_argument("--voice", help="one voice of each engine (default: its first voice)")
    bench.add_argument(
        "--repeats", type=int, default=3, help="warm repetitions per sentence (default 3)"
    )
    bench.add_argument("--out", type=Path, help="bench JSON (default <repo>/build/tts-bench.json)")

    synth = sub.add_parser("synthesize", help="speak one sentence into a WAV file")
    synth.add_argument("text")
    synth.add_argument("--engine", dest="engine")
    synth.add_argument("--voice", dest="voice")
    synth.add_argument("--out", type=Path, required=True)
    synth.add_argument("--no-cache", action="store_true", help="synthesize even when cached")

    sub.add_parser("voices", help="print the engines and voices as JSON")
    phrases = sub.add_parser("phrases", help="export phrases.json without synthesizing")
    phrases.add_argument(
        "--json", type=Path, help="path of phrases.json (default <repo>/build/phrases.json)"
    )
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    from dataclasses import replace

    overrides: dict[str, Any] = {}
    if getattr(args, "cache_dir", None) is not None:
        overrides["cache_dir"] = args.cache_dir.expanduser()
    if getattr(args, "host", None):
        overrides["host"] = args.host
    if getattr(args, "port", None):
        overrides["port"] = args.port
    return replace(settings, **overrides) if overrides else settings


def _registry(settings: Settings) -> EngineRegistry:
    registry = EngineRegistry(
        kokoro_device=settings.kokoro_device,
        xtts_device=settings.xtts_device,
        xtts_speed=settings.xtts_speed,
        default_engine=settings.default_engine,
        default_voice=settings.default_voice,
        edge_rate=settings.edge_rate,
    )
    # The dashboard's last choice wins over the environment, as in the service.
    apply_persisted(registry, DefaultStore(settings.defaults_file))
    return registry


# --------------------------------------------------------------------------- serve


def _serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .service import create_app

    registry = _registry(settings)
    if args.warm:
        log.info("warming %s", settings.default_engine)
        registry.load(settings.default_engine)
    app = create_app(settings=settings, registry=registry)
    log.info(
        "cpsa-tts %s on http://%s:%d (default %s/%s, cache %s)",
        __version__,
        settings.host,
        settings.port,
        settings.default_engine,
        settings.default_voice,
        settings.cache_dir,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())
    return 0


# --------------------------------------------------------------------- pregenerate


def _pregenerate(args: argparse.Namespace, settings: Settings) -> int:
    root = repo_root()
    document = collect(root)
    json_path = args.json or root / "build" / "phrases.json"
    write_phrases(json_path, document)
    counts = document.counts()
    log.info(
        "phrases.json: %d sentences (%d rulebook, %d facts) -> %s",
        counts["total"],
        counts["rulebook"],
        counts["fact"],
        json_path,
    )
    if args.dry_run:
        return 0

    registry = _registry(settings)
    cache = WavCache(settings.cache_dir)
    selected = _selected_engines(registry, args.engines)
    report: dict[str, Any] = {"cache_dir": str(cache.directory), "engines": {}}
    for engine in selected:
        voice = _pick_voice(engine, args.voice)
        rows = document.phrases[: args.limit] if args.limit else document.phrases
        started = time.perf_counter()
        synthesized = 0
        skipped = 0
        durations: list[float] = []
        for index, phrase in enumerate(rows, start=1):
            if not cache.has(engine.id, voice.id, phrase.text):
                began = time.perf_counter()
                samples = engine.synthesize(phrase.text, voice.id)
                cache.write(engine.id, voice.id, phrase.text, samples, engine.sample_rate)
                durations.append(time.perf_counter() - began)
                synthesized += 1
            else:
                skipped += 1
            if index % 50 == 0:
                log.info("%s/%s: %d/%d", engine.id, voice.id, index, len(rows))
        elapsed = time.perf_counter() - started
        report["engines"][f"{engine.id}/{voice.id}"] = {
            "phrases": len(rows),
            "synthesized": synthesized,
            "cached": skipped,
            "seconds": round(elapsed, 1),
            "median_ms": round(statistics.median(durations) * 1000, 1) if durations else None,
        }
        log.info(
            "%s/%s: %d new, %d cached, %.1fs (median %.0f ms/sentence)",
            engine.id,
            voice.id,
            synthesized,
            skipped,
            elapsed,
            (statistics.median(durations) * 1000) if durations else 0.0,
        )

    entries = cache.entries()
    report["cache_entries"] = entries
    report["cache_bytes"] = cache.size_bytes()
    report_path = args.report or root / "build" / "tts-pregenerate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("cache now holds %d WAV files (%.1f MB)", entries, cache.size_bytes() / 1e6)
    log.info("report -> %s", report_path)
    return 0


def _key_for(engine: Engine, voice: Voice, text: str) -> str:
    from .cache import cache_key

    return cache_key(engine.id, voice.id, text)


def _selected_engines(registry: EngineRegistry, requested: list[str] | None) -> list[Engine]:
    if not requested:
        return [registry.default_engine()]
    engines: list[Engine] = []
    for engine_id in requested:
        engines.append(registry.get(engine_id))
    return engines


def _pick_voice(engine: Engine, requested: str | None) -> Voice:
    if requested is None:
        return engine.voices[0]
    for voice in engine.voices:
        if voice.id == requested:
            return voice
    known = ", ".join(voice.id for voice in engine.voices)
    raise EngineError(f"{engine.id} has no voice {requested!r} (known: {known})")


# ------------------------------------------------------------------------- samples


def _samples(args: argparse.Namespace, settings: Settings) -> int:
    registry = _registry(settings)
    engines = (
        list(registry.engines.values())
        if not args.engines
        else _selected_engines(registry, args.engines)
    )
    out_dir = (args.out or settings.samples_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    known = {phrase.text for phrase in collect(repo_root()).phrases}
    for slug, text in SAMPLE_TEXTS:
        if text not in known:
            log.warning("sample %s is no longer a fixed dashboard phrase", slug)
    entries: list[dict[str, Any]] = []
    for engine in engines:
        voices = [voice for voice in engine.voices if not args.voices or voice.id in args.voices]
        if not voices:
            log.warning("%s: no matching voice", engine.id)
            continue
        for voice in voices:
            for slug, text in SAMPLE_TEXTS:
                path = out_dir / f"{engine.id}-{voice.id}-{slug}.wav"
                if path.exists() and not args.force:
                    log.info("skip %s (exists)", path.name)
                    entries.append(
                        {
                            "engine": engine.id,
                            "voice": voice.id,
                            "voice_name": voice.name,
                            "slug": slug,
                            "text": text,
                            "file": path.name,
                            "ms": None,
                        }
                    )
                    continue
                began = time.perf_counter()
                samples = engine.synthesize(text, voice.id)
                elapsed_ms = (time.perf_counter() - began) * 1000
                write_wav_atomic(path, samples, engine.sample_rate)
                entries.append(
                    {
                        "engine": engine.id,
                        "voice": voice.id,
                        "voice_name": voice.name,
                        "slug": slug,
                        "text": text,
                        "file": path.name,
                        "ms": round(elapsed_ms, 1),
                    }
                )
                log.info("%s %s/%s %.0f ms", path.name, engine.id, voice.id, elapsed_ms)
    (out_dir / "index.json").write_text(
        json.dumps(
            {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "samples": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "index.html").write_text(_listen_page(entries), encoding="utf-8")
    log.info("samples -> %s (%d WAV files, open index.html)", out_dir, len(entries))
    return 0


def _listen_page(entries: list[dict[str, Any]]) -> str:
    """A dependency-free listening page: pick the voice by ear."""
    by_text: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_text.setdefault(entry["text"], []).append(entry)
    rows: list[str] = []
    for text, group in by_text.items():
        rows.append(f"<section><h2>{_escape(text)}</h2>")
        for entry in group:
            label = f"{entry['engine']} · {entry['voice_name']} ({entry['voice']})"
            took = f" · {entry['ms']:.0f} ms" if entry.get("ms") is not None else ""
            rows.append(
                f"<p><strong>{_escape(label)}</strong>{took}<br>"
                f'<audio controls preload="none" src="{_escape(entry["file"])}"></audio></p>'
            )
        rows.append("</section>")
    return (
        '<!doctype html>\n<html lang="it"><meta charset="utf-8">'
        "<title>CPSA TTS — ascolto</title>\n"
        "<style>body{font-family:system-ui,sans-serif;max-width:52rem;margin:2rem auto;"
        "padding:0 1rem;background:#0b1020;color:#e8f4ff}section{margin:2rem 0}"
        "h2{font-size:1rem;color:#7fe3ff;font-weight:600}audio{width:100%;margin-top:.4rem}"
        "strong{font-weight:600}p{margin:.6rem 0 1.4rem}</style>\n"
        "<h1>CPSA TTS — ascolto comparato</h1>\n"
        "<p>Kokoro è Apache-2.0 e gira su CPU; XTTS-v2 è CPML (non commerciale) e gira su GPU; "
        "le voci Microsoft arrivano da edge-tts (online, client non ufficiale). "
        "Le stesse cinque frasi, tutte le voci.</p>\n" + "\n".join(rows) + "\n</html>\n"
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


# --------------------------------------------------------------------------- bench


def _bench(args: argparse.Namespace, settings: Settings) -> int:
    registry = _registry(settings)
    engines = (
        list(registry.engines.values())
        if not args.engines
        else _selected_engines(registry, args.engines)
    )
    report: dict[str, Any] = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "engines": {}}
    for engine in engines:
        voice = _pick_voice(engine, args.voice)
        engine.unload()
        began = time.perf_counter()
        engine.load()
        load_ms = (time.perf_counter() - began) * 1000
        cold_times: list[float] = []
        cold_audio: list[float] = []
        for text in BENCH_TEXTS:
            began = time.perf_counter()
            samples = engine.synthesize(text, voice.id)
            cold_times.append((time.perf_counter() - began) * 1000)
            cold_audio.append(len(samples) / engine.sample_rate)
        warm: list[float] = []
        for _ in range(max(1, args.repeats)):
            for text in BENCH_TEXTS:
                began = time.perf_counter()
                engine.synthesize(text, voice.id)
                warm.append((time.perf_counter() - began) * 1000)
        device = engine.device
        gpu_mb = None
        if device == "cuda":
            try:
                import torch

                gpu_mb = round(torch.cuda.max_memory_allocated() / 1e6)
            except Exception:  # pragma: no cover - best effort
                gpu_mb = None
        report["engines"][engine.id] = {
            "voice": voice.id,
            "device": device,
            "load_ms": round(load_ms),
            "cold_ms": round(statistics.mean(cold_times)),
            "cold_per_sentence_ms": [round(value) for value in cold_times],
            "warm_median_ms": round(statistics.median(warm)),
            "warm_min_ms": round(min(warm)),
            "audio_s": round(statistics.mean(cold_audio), 2),
            "rtf_warm": round(statistics.median(warm) / 1000 / statistics.mean(cold_audio), 3),
            "gpu_peak_mb": gpu_mb,
        }
        log.info(
            "%s on %s: load %d ms, cold %.0f ms, warm median %.0f ms (%.0f ms min), RTF %.2f%s",
            engine.id,
            device,
            load_ms,
            statistics.mean(cold_times),
            statistics.median(warm),
            min(warm),
            report["engines"][engine.id]["rtf_warm"],
            f", {gpu_mb} MB VRAM" if gpu_mb else "",
        )
    out = args.out or repo_root() / "build" / "tts-bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("bench -> %s", out)
    return 0


# ---------------------------------------------------------------------- synthesize


def _synthesize(args: argparse.Namespace, settings: Settings) -> int:
    registry = _registry(settings)
    cache = WavCache(settings.cache_dir)
    engine, voice = registry.resolve(args.engine, args.voice)
    payload = None if args.no_cache else cache.read(engine.id, voice.id, args.text)
    if payload is None:
        began = time.perf_counter()
        samples = engine.synthesize(args.text, voice.id)
        elapsed_ms = (time.perf_counter() - began) * 1000
        path = cache.write(engine.id, voice.id, args.text, samples, engine.sample_rate)
        log.info(
            "synthesized %s/%s in %.0f ms (%.2fs audio) -> %s",
            engine.id,
            voice.id,
            elapsed_ms,
            len(samples) / engine.sample_rate,
            path,
        )
    else:
        path = cache.path(_key_for(engine, voice, args.text))
        log.info("cache hit -> %s", path)
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_bytes(path.read_bytes())
    return 0


def _voices(args: argparse.Namespace, settings: Settings) -> int:
    registry = _registry(settings)
    print(
        json.dumps(
            {"defaults": registry.defaults(), "engines": registry.info()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _phrases(args: argparse.Namespace, settings: Settings) -> int:
    document: PhrasesDocument = collect(repo_root())
    path = args.json or repo_root() / "build" / "phrases.json"
    write_phrases(path, document)
    print(json.dumps(document.counts(), ensure_ascii=False))
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
