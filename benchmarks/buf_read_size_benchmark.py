#!/usr/bin/env python3
#
# This file is part of gunicorn released under the MIT license.
# See the NOTICE for more information.

"""
Benchmark request-body processing across different buf_read_size values.

Runs a temporary Gunicorn server for each buf_read_size, uploads a generated
payload, and reports client round-trip time plus server-side body read time.

Usage:
    python benchmarks/buf_read_size_benchmark.py
    python benchmarks/buf_read_size_benchmark.py --buf-sizes 1024,4096,65536
    python benchmarks/buf_read_size_benchmark.py --payload-size 104857600
"""

import argparse
import http.client
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
APP_MODULE = "buf_read_size_benchmark:application"
DEFAULT_BIND = "127.0.0.1:18080"
DEFAULT_BUF_SIZES = [1024, 4096, 65536, 1048576]
DEFAULT_PAYLOAD_SIZE = 100 * 1024 * 1024
DEFAULT_GENERATOR_CHUNK_SIZE = 1024 * 1024


def application(environ, start_response):
    """WSGI app used by the benchmark harness."""
    path = environ.get("PATH_INFO", "/")

    if path == "/health":
        body = b"ok"
        start_response(
            "200 OK",
            [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))],
        )
        return [body]

    if path != "/upload":
        body = b"not found"
        start_response(
            "404 Not Found",
            [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))],
        )
        return [body]

    content_length = int(environ.get("CONTENT_LENGTH") or 0)
    started_at = time.perf_counter()
    body = environ["wsgi.input"].read()
    read_seconds = time.perf_counter() - started_at
    response = json.dumps(
        {
            "expected_bytes": content_length,
            "read_seconds": read_seconds,
            "received_bytes": len(body),
        }
    ).encode("ascii")

    start_response(
        "200 OK",
        [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response))),
            ("X-Body-Read-Duration", f"{read_seconds:.9f}"),
            ("X-Expected-Length", str(content_length)),
            ("X-Received-Length", str(len(body))),
        ],
    )
    return [response]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Gunicorn request-body processing across buf_read_size values."
    )
    parser.add_argument(
        "--buf-sizes",
        default=",".join(str(size) for size in DEFAULT_BUF_SIZES),
        help="Comma-separated buf_read_size values to test.",
    )
    parser.add_argument(
        "--payload-size",
        type=int,
        default=DEFAULT_PAYLOAD_SIZE,
        help="Payload size in bytes. Default: 100 MiB.",
    )
    parser.add_argument(
        "--generator-chunk-size",
        type=int,
        default=DEFAULT_GENERATOR_CHUNK_SIZE,
        help="Chunk size in bytes for the generated request payload.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Measured requests per buf_read_size value.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup requests per buf_read_size value before timing.",
    )
    parser.add_argument(
        "--bind",
        default=DEFAULT_BIND,
        help="Bind address for the temporary Gunicorn instance.",
    )
    parser.add_argument(
        "--worker-class",
        default="sync",
        help="Gunicorn worker class to benchmark.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Gunicorn workers.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Threads per worker.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for benchmark requests.",
    )
    parser.add_argument("--output", help="Output JSON file for results.")
    return parser.parse_args()


def parse_buf_sizes(raw_value):
    values = []

    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue

        value = int(part, 0)
        if value <= 0:
            raise ValueError(f"buf size must be greater than zero: {value}")
        values.append(value)

    if not values:
        raise ValueError("at least one buf size is required")

    return values


def parse_bind(bind):
    host, separator, port = bind.rpartition(":")
    if not separator or not host or not port:
        raise ValueError("bind must be in host:port format")
    return host, int(port)


def validate_args(args):
    if args.payload_size <= 0:
        raise ValueError("payload size must be greater than zero")
    if args.generator_chunk_size <= 0:
        raise ValueError("generator chunk size must be greater than zero")
    if args.iterations <= 0:
        raise ValueError("iterations must be greater than zero")
    if args.warmup < 0:
        raise ValueError("warmup must be zero or greater")


def wait_for_server(bind, timeout=10.0):
    host, port = parse_bind(bind)
    deadline = time.monotonic() + timeout
    last_error = None

    while time.monotonic() < deadline:
        connection = None
        try:
            connection = http.client.HTTPConnection(host, port, timeout=1)
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
        finally:
            if connection is not None:
                connection.close()

    raise RuntimeError(f"Gunicorn did not become ready on {bind}: {last_error}")


def start_gunicorn(args, buf_size):
    cmd = [
        sys.executable,
        "-m",
        "gunicorn",
        "--bind",
        args.bind,
        "--worker-class",
        args.worker_class,
        "--workers",
        str(args.workers),
        "--threads",
        str(args.threads),
        "--buf-read-size",
        str(buf_size),
        "--access-logfile",
        "/dev/null",
        "--error-logfile",
        "-",
        "--log-level",
        "warning",
        APP_MODULE,
    ]

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([str(BENCHMARK_DIR), pythonpath])
    else:
        env["PYTHONPATH"] = str(BENCHMARK_DIR)

    proc = subprocess.Popen(
        cmd,
        cwd=BENCHMARK_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        wait_for_server(args.bind)
    except Exception:
        stop_gunicorn(proc)
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        raise RuntimeError(
            f"Failed to start gunicorn for buf_read_size={buf_size}\n{stderr}"
        )

    return proc


def stop_gunicorn(proc):
    if proc.poll() is not None:
        return

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def stream_post(bind, payload_size, generator_chunk_size, timeout):
    host, port = parse_bind(bind)
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    chunk = b"x" * generator_chunk_size
    started_at = time.perf_counter()

    connection.putrequest("POST", "/upload")
    connection.putheader("Content-Type", "application/octet-stream")
    connection.putheader("Content-Length", str(payload_size))
    connection.endheaders()

    remaining = payload_size
    while remaining > 0:
        send_size = min(generator_chunk_size, remaining)
        connection.send(chunk[:send_size])
        remaining -= send_size

    response = connection.getresponse()
    body = response.read()
    round_trip_seconds = time.perf_counter() - started_at
    headers = dict(response.getheaders())
    connection.close()

    if response.status != 200:
        raise RuntimeError(f"unexpected status {response.status}: {body[:200]!r}")

    received_length = int(headers["X-Received-Length"])
    expected_length = int(headers["X-Expected-Length"])
    if received_length != payload_size or expected_length != payload_size:
        raise RuntimeError(
            "server did not receive the expected payload size: "
            f"received={received_length} expected={expected_length}"
        )

    payload_mib = payload_size / (1024 * 1024)
    server_read_seconds = float(headers["X-Body-Read-Duration"])
    return {
        "client_mib_per_second": payload_mib / round_trip_seconds,
        "payload_bytes": payload_size,
        "payload_mib": payload_mib,
        "round_trip_seconds": round_trip_seconds,
        "server_mib_per_second": payload_mib / server_read_seconds,
        "server_read_seconds": server_read_seconds,
    }


def summarize_runs(runs):
    round_trip = [run["round_trip_seconds"] for run in runs]
    server_read = [run["server_read_seconds"] for run in runs]
    client_throughput = [run["client_mib_per_second"] for run in runs]
    server_throughput = [run["server_mib_per_second"] for run in runs]

    return {
        "client_mib_per_second_mean": statistics.fmean(client_throughput),
        "round_trip_seconds_mean": statistics.fmean(round_trip),
        "round_trip_seconds_median": statistics.median(round_trip),
        "server_mib_per_second_mean": statistics.fmean(server_throughput),
        "server_read_seconds_mean": statistics.fmean(server_read),
        "server_read_seconds_median": statistics.median(server_read),
    }


def print_configuration(args, buf_sizes):
    print("buf_read_size benchmark")
    print("=" * 60)
    print(f"Bind: {args.bind}")
    print(f"Worker class: {args.worker_class}")
    print(f"Workers: {args.workers}")
    print(f"Threads: {args.threads}")
    print(f"Payload size: {args.payload_size} bytes")
    print(f"Generator chunk size: {args.generator_chunk_size} bytes")
    print(f"Warmup: {args.warmup}")
    print(f"Iterations: {args.iterations}")
    print(f"buf_read_size values: {', '.join(str(size) for size in buf_sizes)}")


def print_summary(buf_size, summary):
    print(
        f"  buf_read_size={buf_size:<8d} "
        f"server_read_mean={summary['server_read_seconds_mean']:.4f}s "
        f"round_trip_mean={summary['round_trip_seconds_mean']:.4f}s "
        f"server_throughput={summary['server_mib_per_second_mean']:.2f} MiB/s"
    )


def run_case(args, buf_size):
    proc = start_gunicorn(args, buf_size)

    try:
        for _ in range(args.warmup):
            stream_post(
                args.bind,
                args.payload_size,
                args.generator_chunk_size,
                args.timeout,
            )

        runs = []
        for iteration in range(args.iterations):
            metrics = stream_post(
                args.bind,
                args.payload_size,
                args.generator_chunk_size,
                args.timeout,
            )
            runs.append(metrics)
            print(
                f"  Run {iteration + 1}/{args.iterations}: "
                f"server_read={metrics['server_read_seconds']:.4f}s "
                f"round_trip={metrics['round_trip_seconds']:.4f}s"
            )

        return {
            "runs": runs,
            "summary": summarize_runs(runs),
        }
    finally:
        stop_gunicorn(proc)


def write_results(output_path, results):
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as fileobj:
        json.dump(results, fileobj, indent=2)
    print(f"\nResults saved to {path}")


def main():
    args = parse_args()
    validate_args(args)
    buf_sizes = parse_buf_sizes(args.buf_sizes)

    print_configuration(args, buf_sizes)

    results = {
        "config": {
            "bind": args.bind,
            "buf_sizes": buf_sizes,
            "generator_chunk_size": args.generator_chunk_size,
            "iterations": args.iterations,
            "payload_size": args.payload_size,
            "threads": args.threads,
            "warmup": args.warmup,
            "worker_class": args.worker_class,
            "workers": args.workers,
        },
        "results": {},
    }

    print("\nResults")
    print("-" * 60)
    for buf_size in buf_sizes:
        print(f"\nRunning buf_read_size={buf_size}")
        case_result = run_case(args, buf_size)
        results["results"][str(buf_size)] = case_result
        print_summary(buf_size, case_result["summary"])

    print("\nSummary")
    print("-" * 60)
    for buf_size in buf_sizes:
        print_summary(buf_size, results["results"][str(buf_size)]["summary"])

    if args.output:
        write_results(args.output, results)


if __name__ == "__main__":
    main()
