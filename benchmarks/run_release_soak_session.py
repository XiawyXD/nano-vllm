"""Own one localhost server, run smoke + soak, then clean up only our children.

Example: CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python benchmarks/run_release_soak_session.py
         --output artifacts/release_soak_session_20260920 --duration 1800
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_session(path, session):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def ensure_port_available(host, port):
    """Fail before starting anything if another process already owns the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))


def wait_ready(child, url, log_path, timeout):
    deadline = time.monotonic() + timeout
    marker = f'nano-vLLM HTTP/SSE listening on {url}'
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f'our server exited before readiness: {child.returncode}')
        # A health response alone could come from a process that raced our bind.
        # Require our own server log's post-bind startup marker as well.
        if marker in log_path.read_text(encoding='utf-8', errors='replace'):
            try:
                with urllib.request.urlopen(url + '/health', timeout=2) as response:
                    data = json.load(response)
                    if response.status == 200 and data.get('status') == 'ok':
                        if child.poll() is None:
                            return
            except (OSError, urllib.error.URLError, ValueError):
                pass
        time.sleep(0.5)
    raise TimeoutError(f'our server did not become ready within {timeout}s')


def stop_owned_child(name, child, record):
    """Never discover/kill a PID by port or name; only signal this Popen object."""
    record['cleanup_started_at'] = timestamp()
    record['signals'] = []
    if child.poll() is None:
        for label, action, timeout in (
            ('SIGINT', lambda: child.send_signal(signal.SIGINT), 30),
            ('terminate', child.terminate, 10),
            ('kill', child.kill, 10),
        ):
            if child.poll() is not None:
                break
            try:
                action()
                record['signals'].append(label)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                continue
    record['returncode'] = child.poll()
    record['reaped'] = child.poll() is not None
    record['cleanup_finished_at'] = timestamp()
    print(json.dumps({'child': name, 'pid': child.pid, 'cleanup': record}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New session directory; existing data is never overwritten')
    parser.add_argument('--model', default='/home/hrxu/huggingface/Qwen3-1.7B')
    parser.add_argument('--port', type=int, default=18762)
    parser.add_argument('--duration', type=float, default=1800, help='Formal soak duration; use 1800 for release acceptance')
    parser.add_argument('--ready-timeout', type=float, default=600)
    args = parser.parse_args()
    if (not (1 <= args.port <= 65535) or not math.isfinite(args.duration)
            or not math.isfinite(args.ready_timeout) or args.duration <= 0 or args.ready_timeout <= 0):
        parser.error('port must be 1..65535; durations must be positive')
    host = '127.0.0.1'
    ensure_port_available(host, args.port)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    session_path = output / 'session.json'
    env = dict(os.environ)  # Preserve CUDA_VISIBLE_DEVICES and distributed port settings.
    env['NANOVLLM_SPEC_GRAPH'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    url = f'http://{host}:{args.port}'
    server_command = [sys.executable, str(root/'serve_nano_vllm.py'),
        '--model', args.model, '--host', host, '--port', str(args.port),
        '--max-model-len', '1536', '--max-num-seqs', '16',
        '--max-num-batched-tokens', '512', '--prefill-chunk-size', '256',
        '--max-prefill-steps', '1', '--gpu-memory-utilization', '0.5',
        '--sampler-backend', 'torch']
    session = dict(status='RUNNING', started_at=timestamp(), orchestrator_pid=os.getpid(),
        args=vars(args), cwd=str(root), server_url=url,
        environment={key: env.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'MASTER_ADDR', 'MASTER_PORT',
            'NANOVLLM_SPEC_GRAPH', 'HF_HUB_OFFLINE')},
        server_command=server_command, children={}, stages={},
        release_duration_satisfied=args.duration >= 1800,
        note='Only this script\'s Popen children are stopped; prefix caching remains enabled.')
    children = []
    write_session(session_path, session)
    try:
        with ExitStack() as stack:
            server_log = stack.enter_context((output/'server.log').open('w', encoding='utf-8'))
            server = subprocess.Popen(server_command, cwd=root, env=env,
                                      stdout=server_log, stderr=subprocess.STDOUT)
            session['children']['server'] = dict(pid=server.pid, command=server_command)
            children.append(('server', server))
            write_session(session_path, session)
            try:
                wait_ready(server, url, output/'server.log', args.ready_timeout)
                session['server_ready_at'] = timestamp()
                write_session(session_path, session)
                for name, duration, concurrency in (('smoke', 15, 2), ('soak', args.duration, 8)):
                    command = [sys.executable, str(root/'benchmarks/release_soak.py'),
                        '--base-url', url, '--output', str(output/name),
                        '--duration', str(duration), '--concurrency', str(concurrency),
                        '--ngram-size', '3', '--cancel-every', '4']
                    stage_log = stack.enter_context((output/f'{name}.log').open('w', encoding='utf-8'))
                    stage = dict(command=command, started_at=timestamp(),
                                 duration=duration, concurrency=concurrency)
                    session['stages'][name] = stage
                    child = subprocess.Popen(command, cwd=root, env=env,
                                             stdout=stage_log, stderr=subprocess.STDOUT)
                    children.append((name, child))
                    session['children'][name] = dict(pid=child.pid, command=command)
                    write_session(session_path, session)
                    print(json.dumps({'stage': name, 'pid': child.pid, 'duration': duration}), flush=True)
                    while child.poll() is None:
                        if server.poll() is not None:
                            raise RuntimeError(f'our server exited during {name}: {server.returncode}')
                        time.sleep(0.5)
                    stage['returncode'] = child.returncode
                    stage['finished_at'] = timestamp()
                    summary_path = output/name/'summary.json'
                    stage['summary_path'] = str(summary_path)
                    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                    stage['summary_status'] = summary.get('status', 'MISSING')
                    write_session(session_path, session)
                    if child.returncode != 0 or stage['summary_status'] != 'PASS':
                        raise RuntimeError(f'{name} failed: exit={child.returncode}, status={stage["summary_status"]}')
                session['status'] = 'PASS'
            finally:
                # Reverse order means stop a still-running client before its server.
                for name, child in reversed(children):
                    stop_owned_child(name, child, session['children'][name])
                    write_session(session_path, session)
    except BaseException as exc:
        session['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
        session['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        session['finished_at'] = timestamp()
        session['all_owned_children_reaped'] = all(child.poll() is not None for _, child in children)
        if not session['all_owned_children_reaped']:
            session['status'] = 'FAIL'
            session['cleanup_error'] = 'one or more owned children could not be reaped'
        try:
            ensure_port_available(host, args.port)
            session['port_available_after_cleanup'] = True
        except OSError:
            session['port_available_after_cleanup'] = False
        write_session(session_path, session)
        print(json.dumps({'status': session['status'], 'session': str(session_path),
                          'children_reaped': session['all_owned_children_reaped']}), flush=True)


if __name__ == '__main__':
    main()
