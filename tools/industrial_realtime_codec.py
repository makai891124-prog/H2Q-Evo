import argparse
import concurrent.futures
import hashlib
import json
import time
import zlib
from pathlib import Path
from typing import Dict, List


MAGIC = b"H2QZ1\n"
DEFAULT_CHUNK = 1024 * 256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(DEFAULT_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compress_file(src: Path, dst: Path, level: int = 6, chunk_size: int = DEFAULT_CHUNK) -> Dict[str, float]:
    src_size = src.stat().st_size
    t0 = time.perf_counter()

    comp = zlib.compressobj(level)
    sha = hashlib.sha256()

    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(MAGIC)
        header_pos = fout.tell()
        fout.write((0).to_bytes(8, "big"))
        header_bytes = b""

        while True:
            block = fin.read(chunk_size)
            if not block:
                break
            sha.update(block)
            payload = comp.compress(block)
            if payload:
                fout.write(payload)

        tail = comp.flush()
        if tail:
            fout.write(tail)

        meta = {
            "source_name": src.name,
            "source_size": src_size,
            "source_sha256": sha.hexdigest(),
            "codec": "zlib",
            "level": level,
            "chunk_size": chunk_size,
            "created_at": int(time.time()),
        }
        header_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        fout.write(header_bytes)

        end_pos = fout.tell()
        fout.seek(header_pos)
        fout.write(len(header_bytes).to_bytes(8, "big"))
        fout.seek(end_pos)

    dt = time.perf_counter() - t0
    out_size = dst.stat().st_size
    return {
        "input_bytes": float(src_size),
        "output_bytes": float(out_size),
        "ratio": float(src_size / max(out_size, 1)),
        "compress_seconds": float(dt),
        "compress_mb_s": float((src_size / (1024 * 1024)) / max(dt, 1e-12)),
    }


def decompress_file(src: Path, dst: Path, chunk_size: int = DEFAULT_CHUNK) -> Dict[str, float]:
    t0 = time.perf_counter()

    with src.open("rb") as fin:
        magic = fin.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Invalid H2QZ magic header")
        header_len = int.from_bytes(fin.read(8), "big")
        compressed_blob = fin.read()

    if header_len <= 0 or header_len > len(compressed_blob):
        raise ValueError("Invalid metadata header length")

    header_bytes = compressed_blob[-header_len:]
    payload = compressed_blob[:-header_len]
    meta = json.loads(header_bytes.decode("utf-8"))

    decomp = zlib.decompressobj()
    sha = hashlib.sha256()

    with dst.open("wb") as fout:
        cursor = 0
        while cursor < len(payload):
            part = payload[cursor : cursor + chunk_size]
            cursor += len(part)
            out = decomp.decompress(part)
            if out:
                fout.write(out)
                sha.update(out)
        tail = decomp.flush()
        if tail:
            fout.write(tail)
            sha.update(tail)

    dt = time.perf_counter() - t0
    out_size = dst.stat().st_size
    actual_sha = sha.hexdigest()
    expected_sha = meta.get("source_sha256", "")
    if actual_sha != expected_sha:
        raise ValueError("Checksum mismatch after decompression")

    return {
        "output_bytes": float(out_size),
        "decompress_seconds": float(dt),
        "decompress_mb_s": float((out_size / (1024 * 1024)) / max(dt, 1e-12)),
        "checksum_match": True,
    }


def benchmark_on_files(files: List[Path], out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for src in files:
        comp = out_dir / f"{src.name}.h2qz"
        dec = out_dir / f"{src.name}.roundtrip"

        c = compress_file(src, comp, level=6)
        d = decompress_file(comp, dec)

        rows.append(
            {
                "file": str(src),
                "input_bytes": c["input_bytes"],
                "compressed_bytes": c["output_bytes"],
                "ratio": c["ratio"],
                "compress_mb_s": c["compress_mb_s"],
                "decompress_mb_s": d["decompress_mb_s"],
                "checksum_match": d["checksum_match"],
            }
        )

    ratios = [r["ratio"] for r in rows]
    cspd = [r["compress_mb_s"] for r in rows]
    dspd = [r["decompress_mb_s"] for r in rows]

    return {
        "rows": rows,
        "summary": {
            "file_count": len(rows),
            "mean_ratio": float(sum(ratios) / max(len(ratios), 1)),
            "mean_compress_mb_s": float(sum(cspd) / max(len(cspd), 1)),
            "mean_decompress_mb_s": float(sum(dspd) / max(len(dspd), 1)),
            "all_checksum_match": bool(all(r["checksum_match"] for r in rows)),
        },
    }


def _destination_for(src: Path, input_dir: Path, output_dir: Path, mode: str) -> Path:
    rel = src.relative_to(input_dir)
    if mode == "compress":
        return output_dir / f"{rel}.h2qz"
    if src.suffix != ".h2qz":
        raise ValueError(f"Cannot decompress non-.h2qz file: {src}")
    stem_name = src.name[: -len(".h2qz")]
    return output_dir / rel.with_name(stem_name)


def _discover_files(input_dir: Path, mode: str, recursive: bool, pattern: str) -> List[Path]:
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    files = []
    for p in iterator:
        if not p.is_file():
            continue
        if mode == "compress" and p.suffix == ".h2qz":
            continue
        if mode == "decompress" and p.suffix != ".h2qz":
            continue
        files.append(p)
    return sorted(files)


def _convert_one(task: Dict[str, object]) -> Dict[str, object]:
    src = Path(task["src"])
    dst = Path(task["dst"])
    mode = str(task["mode"])
    level = int(task.get("level", 6))
    chunk_size = int(task.get("chunk_size", DEFAULT_CHUNK))
    started = time.time()

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "compress":
        stats = compress_file(src, dst, level=level, chunk_size=chunk_size)
    else:
        stats = decompress_file(src, dst, chunk_size=chunk_size)

    return {
        "mode": mode,
        "src": str(src),
        "dst": str(dst),
        "started_at": int(started),
        "finished_at": int(time.time()),
        "stats": stats,
        "ok": True,
    }


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    workers: int = 4,
    recursive: bool = True,
    pattern: str = "*",
    level: int = 6,
    chunk_size: int = DEFAULT_CHUNK,
) -> Dict[str, object]:
    files = _discover_files(input_dir, mode=mode, recursive=recursive, pattern=pattern)
    tasks = []
    for src in files:
        dst = _destination_for(src, input_dir=input_dir, output_dir=output_dir, mode=mode)
        tasks.append(
            {
                "mode": mode,
                "src": str(src),
                "dst": str(dst),
                "level": level,
                "chunk_size": chunk_size,
            }
        )

    rows = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = [pool.submit(_convert_one, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            rows.append(fut.result())

    elapsed = time.perf_counter() - t0
    ok_count = sum(1 for r in rows if r.get("ok"))
    return {
        "mode": mode,
        "workers": int(max(workers, 1)),
        "task_count": len(tasks),
        "ok_count": ok_count,
        "failed_count": len(tasks) - ok_count,
        "elapsed_seconds": float(elapsed),
        "rows": rows,
    }


def watch_directory(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    workers: int = 4,
    recursive: bool = True,
    pattern: str = "*",
    level: int = 6,
    interval: float = 1.5,
    max_cycles: int = 0,
    chunk_size: int = DEFAULT_CHUNK,
) -> Dict[str, object]:
    seen_mtime: Dict[str, float] = {}
    processed = 0
    rows: List[Dict[str, object]] = []
    started = int(time.time())
    cycles = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        pending: Dict[str, concurrent.futures.Future] = {}
        while True:
            files = _discover_files(input_dir, mode=mode, recursive=recursive, pattern=pattern)
            for src in files:
                src_key = str(src)
                mtime = src.stat().st_mtime
                changed = src_key not in seen_mtime or mtime > seen_mtime[src_key]
                if not changed:
                    continue
                if src_key in pending and not pending[src_key].done():
                    continue

                dst = _destination_for(src, input_dir=input_dir, output_dir=output_dir, mode=mode)
                task = {
                    "mode": mode,
                    "src": str(src),
                    "dst": str(dst),
                    "level": level,
                    "chunk_size": chunk_size,
                }
                pending[src_key] = pool.submit(_convert_one, task)
                seen_mtime[src_key] = mtime

            done_keys = [k for k, f in pending.items() if f.done()]
            for key in done_keys:
                result = pending[key].result()
                rows.append(result)
                processed += 1
                del pending[key]

            cycles += 1
            if max_cycles > 0 and cycles >= max_cycles:
                break
            time.sleep(max(interval, 0.05))

        for key, fut in list(pending.items()):
            result = fut.result()
            rows.append(result)
            processed += 1
            del pending[key]

    return {
        "mode": mode,
        "workers": int(max(workers, 1)),
        "watch_started_at": started,
        "watch_finished_at": int(time.time()),
        "scan_cycles": int(cycles),
        "processed_count": int(processed),
        "rows": rows,
    }


def _cli() -> None:
    p = argparse.ArgumentParser(description="Industrial realtime compression/decompression converter")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--level", type=int, default=6)

    d = sub.add_parser("decompress")
    d.add_argument("src")
    d.add_argument("dst")

    b = sub.add_parser("benchmark")
    b.add_argument("out_dir")
    b.add_argument("files", nargs="+")

    q = sub.add_parser("batch")
    q.add_argument("input_dir")
    q.add_argument("output_dir")
    q.add_argument("--mode", choices=["compress", "decompress"], default="compress")
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--pattern", default="*")
    q.add_argument("--recursive", action="store_true")
    q.add_argument("--level", type=int, default=6)

    w = sub.add_parser("watch")
    w.add_argument("input_dir")
    w.add_argument("output_dir")
    w.add_argument("--mode", choices=["compress", "decompress"], default="compress")
    w.add_argument("--workers", type=int, default=4)
    w.add_argument("--pattern", default="*")
    w.add_argument("--recursive", action="store_true")
    w.add_argument("--level", type=int, default=6)
    w.add_argument("--interval", type=float, default=1.5)
    w.add_argument("--max-cycles", type=int, default=0)

    args = p.parse_args()
    if args.cmd == "compress":
        res = compress_file(Path(args.src), Path(args.dst), level=args.level)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "decompress":
        res = decompress_file(Path(args.src), Path(args.dst))
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "benchmark":
        files = [Path(x) for x in args.files]
        res = benchmark_on_files(files, Path(args.out_dir))
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "batch":
        res = batch_convert(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            mode=args.mode,
            workers=args.workers,
            recursive=args.recursive,
            pattern=args.pattern,
            level=args.level,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "watch":
        res = watch_directory(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            mode=args.mode,
            workers=args.workers,
            recursive=args.recursive,
            pattern=args.pattern,
            level=args.level,
            interval=args.interval,
            max_cycles=args.max_cycles,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
