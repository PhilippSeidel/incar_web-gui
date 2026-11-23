"""File and process operations on the PC that runs incar.

The GUI imports this module for the local PC and sends its source over SSH for a remote PC
(`<incar python> -c <this source> <op> <json args>`), so both take the same code path.
Only the standard library is imported at module level: numpy, h5py and PIL are imported by the ops that need them.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATASETS_DIR = 'datasets'  # incar.ws_checker.FileType.DATASET
POLICIES_DIR = 'policies'  # incar.ws_checker.FileType.POLICY
RUNS_DIR = 'training_runs'


def _path(path: str) -> Path:
    return Path(os.path.expanduser(path))


def ping() -> dict:
    return {'python': sys.executable, 'incar': (Path(sys.executable).parent / 'incar').is_file()}


def read_text(path: str) -> str:
    return _path(path).read_text()


def write_text(path: str, text: str) -> None:
    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.tmp')  # never leave a half-written file behind
    tmp.write_text(text)
    tmp.replace(target)


def exists(path: str) -> bool:
    return _path(path).exists()


def mkdir(path: str) -> None:
    _path(path).mkdir(parents=True, exist_ok=True)


def list_names(path: str, suffix: str = '') -> list[str]:
    folder = _path(path)
    return sorted(p.name for p in folder.iterdir() if p.name.endswith(suffix)) if folder.is_dir() else []


def read_file(path: str) -> str:
    return base64.b64encode(_path(path).read_bytes()).decode()


def datasets(workspace: str) -> dict:
    """Recorded datasets of a workspace: name -> features from its dataset_config.json."""
    result = {}
    for name in list_names(f'{workspace}/{DATASETS_DIR}'):
        try:
            result[name] = json.loads(_path(f'{workspace}/{DATASETS_DIR}/{name}/dataset_config.json').read_text())['features']
        except (OSError, ValueError, KeyError):
            continue
    return result


def remove_dataset(workspace: str, name: str) -> None:
    """Delete a recorded dataset: only a real dataset folder directly in datasets/; a symlinked one loses its link."""
    if name in ('', '.', '..') or '/' in name:
        raise ValueError(f'not a dataset name: {name!r}')
    folder = _path(f'{workspace}/{DATASETS_DIR}/{name}')
    if not (folder / 'dataset_config.json').is_file():
        raise FileNotFoundError(f'{folder} is not a dataset')
    folder.unlink() if folder.is_symlink() else shutil.rmtree(folder)


def remove_demo(dataset_dir: str, name: str) -> None:
    """Delete one saved demo (demo_<n> with its meta.json) of a dataset; incar numbers new demos after the highest."""
    if not re.fullmatch(r'demo_\d+', name):
        raise ValueError(f'not a demo name: {name!r}')
    folder = _path(f'{dataset_dir}/{name}')
    if folder.is_symlink() or not (folder / 'meta.json').is_file():
        raise FileNotFoundError(f'{folder} is not a saved demo')
    shutil.rmtree(folder)


def _demo_number(name: str) -> int:
    suffix = name.rsplit('_', 1)[-1]
    return int(suffix) if suffix.isdigit() else 1 << 30


def list_demos(dataset_dir: str) -> list[dict]:
    """Complete demos, in recording order. incar writes meta.json last, so a demo without it is still being saved."""
    from datetime import datetime
    demos = []
    for folder in _path(dataset_dir).glob('demo_*'):
        try:
            meta = json.loads((folder / 'meta.json').read_text())
            start, end = datetime.fromisoformat(meta['t_start']), datetime.fromisoformat(meta['t_end'])
        except (OSError, ValueError, KeyError):
            continue
        demos.append({'name': folder.name, 'start': start.isoformat(), 'duration': (end - start).total_seconds(),
                      'takeover': bool(meta.get('is_takeover'))})
    return sorted(demos, key=lambda d: _demo_number(d['name']))


def load_series(feature_dir: str) -> dict:
    """Seconds since demo start and samples of a non-video feature; incar stores timestamps in microseconds."""
    import h5py
    with h5py.File(_path(feature_dir) / 'data.h5', 'r') as f:
        t, data = f['timestamps'][:] / 1e6, f['data'][:]
    if data.dtype.kind in 'OSU':
        return {'t': t.tolist(), 'text': [v.decode() if isinstance(v, bytes) else str(v) for v in data]}
    return {'t': t.tolist(), 'data': data.reshape(len(t), -1).tolist()}


def write_text_series(feature_dir: str, texts: list) -> None:
    """Replace the values of a text feature (e.g. the prompt), one per timestamp. The recorded original is kept once
    as data.h5.orig; the new file is written as a copy first, so a failure never leaves a half-written one."""
    import h5py
    path = _path(feature_dir) / 'data.h5'
    with h5py.File(path, 'r') as f:
        info = h5py.check_string_dtype(f['data'].dtype)
        count = len(f['timestamps'])
    if info is None:
        raise ValueError(f'{path} holds no text')
    if not isinstance(texts, list) or len(texts) != count or not all(isinstance(t, str) for t in texts):
        raise ValueError(f'expected {count} texts, one per timestamp')
    backup = path.with_name('data.h5.orig')
    if not backup.exists():
        shutil.copy2(path, backup)
    tmp = path.with_name('data.h5.tmp')
    shutil.copy2(path, tmp)
    try:
        with h5py.File(tmp, 'r+') as f:
            del f['data']
            f.create_dataset('data', data=texts, dtype=h5py.string_dtype(encoding=info.encoding))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def frame_times(feature_dir: str) -> list[float]:
    """Capture time (s since demo start) of each video frame; frame i is shown at i / video_fps in the mp4."""
    try:
        import h5py
        with h5py.File(_path(feature_dir) / 'timestamps.h5', 'r') as f:
            return (f['timestamps'][:] / 1e6).round(4).tolist()
    except (OSError, KeyError):
        return []  # the cursor then falls back to the plain video time


# ─── training processes ───────────────────────────────────────────────────────────────────────────────────────

def spawn(argv: list[str], log_path: str, cwd: str, env: dict | None = None) -> int:
    """Start a process in its own session, writing to a log file, so it outlives the GUI and the SSH session.
    A wrapping shell records the exit code next to the log (<log>.exit)."""
    log = _path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'wb') as out:  # the child keeps its own copy of the handle
        # trap with a no-op (not ''): the shell survives Ctrl+C to record the code, the training still gets it
        proc = subprocess.Popen(['sh', '-c', 'trap : INT; "$0" "$@"; echo $? > "$INCAR_GUI_EXIT"',
                                 *[os.path.expanduser(a) for a in argv]],
                                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=_path(cwd),
                                env={**os.environ, **(env or {}), 'INCAR_GUI_EXIT': f'{log}.exit'},
                                start_new_session=True)
    return proc.pid


def alive(pid: int) -> bool:
    try:
        os.waitpid(pid, os.WNOHANG)  # reap it if it is our own child (local GUI), else it lingers as a zombie
    except ChildProcessError:
        pass
    try:  # a zombie has an empty cmdline
        return b'train' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return False


def tail(log_path: str, offset: int, size: int, pid: int) -> dict:
    """New log bytes from offset, whether the process still runs, and its exit code once known."""
    running = alive(pid)  # before reading, so no output written before the end is missed
    try:
        with open(_path(log_path), 'rb') as log:
            log.seek(offset)
            data = log.read(size)
    except OSError:
        data = b''
    try:
        code = int(_path(f'{log_path}.exit').read_text())
    except (OSError, ValueError):
        code = None
    return {'data': base64.b64encode(data).decode(), 'alive': running, 'exit': code}


def kill(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def runs(workspace: str) -> list[dict]:
    """Recorded trainings of a workspace, newest first, with whether each is still running."""
    result = []
    for name in reversed(list_names(f'{workspace}/{RUNS_DIR}', '.json')):
        try:
            info = json.loads(_path(f'{workspace}/{RUNS_DIR}/{name}').read_text())
        except (OSError, ValueError):
            continue
        try:  # when an ended training last wrote its log
            modified = _path(info['log_path']).stat().st_mtime
        except (OSError, KeyError):
            modified = None
        policy = info.get('policy', '').rsplit('/', 1)[-1]
        if not policy:  # run files from older GUI versions only name the config
            try:
                policy = json.loads(_path(info['config_path']).read_text())['model_save_name']
            except (OSError, ValueError, KeyError):
                pass
        result.append({**info, 'alive': alive(info['pid']), 'modified': modified, 'policy_name': policy})
    return result


def trained_policies(workspace: str) -> list[dict]:
    """What a training leaves in policies/<name>, all there is of one run outside the GUI (no log): checkpoint
    steps with their times, the best checkpoint's step, the training config and incar's loss plot, if any."""
    result = []
    for name in list_names(f'{workspace}/{POLICIES_DIR}'):
        folder = _path(f'{workspace}/{POLICIES_DIR}/{name}')
        checkpoints = folder / 'checkpoints'
        found = sorted((int(d.name), d.name, d.stat().st_mtime) for d in checkpoints.glob('*') if d.name.isdigit())
        if not found:
            continue

        def read(path: str):
            try:
                return json.loads((checkpoints / path).read_text())
            except (OSError, ValueError):
                return None
        config = read(f'{found[-1][1]}/model/train_config.json') or read('best/model/train_config.json') or {}
        plot = folder / 'losses.svg'
        result.append({'name': name, 'folder': str(folder), 'checkpoints': [[s, t] for s, _, t in found],
                       'best': (read('best/training_state/training_step.json') or {}).get('step'), 'config': config,
                       'plot': str(plot) if plot.is_file() and plot.stat().st_size else '', 'modified': found[-1][2]})
    return result


# ─── camera frames for a remote GUI ───────────────────────────────────────────────────────────────────────────

def pump(streams: list, interval: float = 1 / 30) -> None:
    """Stream changed camera frames from incar's shared memory as JPEG: a JSON header line, then the bytes.
    Runs until the GUI closes the connection."""
    import io
    from multiprocessing import shared_memory
    import numpy as np
    from PIL import Image
    out = sys.stdout.buffer
    maps, last, changed = {}, {}, {}
    while True:
        for name, h, w in streams:
            try:
                if name not in maps:
                    try:
                        maps[name] = shared_memory.SharedMemory(f'incar_{name}', track=False)  # never unlink incar's
                    except TypeError:  # Python < 3.13
                        maps[name] = shared_memory.SharedMemory(f'incar_{name}')
                if maps[name].size < h * w * 3:
                    continue
                frame = np.ndarray((h, w, 3), np.uint8, maps[name].buf).copy()
            except FileNotFoundError:
                continue
            now = time.monotonic()
            if name in last and np.array_equal(last[name], frame):
                if now - changed[name] > 2:
                    maps.pop(name, None)  # re-open: a restarted camera process creates a new segment
                continue
            last[name], changed[name] = frame, now
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, 'JPEG', quality=70)
            data = buffer.getvalue()
            out.write(json.dumps({'name': name, 'size': len(data)}).encode() + b'\n' + data)
            out.flush()
        time.sleep(interval)


OPS = {f.__name__: f for f in (ping, read_text, write_text, exists, mkdir, list_names, read_file, datasets, list_demos,
                                load_series, write_text_series, frame_times, remove_dataset, remove_demo, spawn, tail, kill, runs, trained_policies)}

if __name__ == '__main__':
    op, args = sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
    if op == 'pump':
        try:
            pump(*args)
        except (BrokenPipeError, KeyboardInterrupt):
            pass
        sys.exit()
    try:
        result = {'ok': OPS[op](*args)}
    except Exception as e:  # reported to the GUI instead of a traceback on stderr
        result = {'error': f'{type(e).__name__}: {e}'}
    sys.stdout.write('\n@@JSON@@' + json.dumps(result))
