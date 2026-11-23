"""NiceGUI front end for the incar core.

Setup (own venv, so nicegui's uvloop/httptools don't change how incar's uvicorn runs):
    python3.13 -m venv .venv && .venv/bin/pip install nicegui segno
    echo <incar venv>/lib/python3.13/site-packages > .venv/lib/python3.13/site-packages/incar_venv.pth
Run: `incar run --headless`, then `.venv/bin/python app.py` and open http://127.0.0.1:8090.
A remote incar PC is chosen in Settings (SSH). TRAIN_SIMULATION=1 (optionally SIM_SECONDS=30) makes "Start
training" run sim_train.py, which only prints a realistic training log.
"""
import asyncio
import base64
import hashlib
import io
import itertools
import json
import os
import tempfile
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
import segno
from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from nicegui import app, background_tasks, ui

import agent
import headset
import hosts
import training
import training_ui
from widgets import PILL_STATES, card, heading, icon_button, pill, set_pill, set_props, set_style
from incar_networking.messages.core_status_pb2 import ConnectionState, CoreStatus
from incar_networking.messages.teleop_command_pb2 import TeleopCommand
from incar_networking.webrtc.webrtc_connection import WebRTCConnection

CORE_URL = os.getenv('INCAR_URL', 'http://127.0.0.1:8000')
API_KEY = os.getenv('INCAR_API_KEY', 'incar_admin')  # incar's default key for local (non-headless) runs

client = httpx.AsyncClient(base_url=CORE_URL, headers={'X-API-Key': API_KEY}, timeout=5)

# Shared by all browser tabs: /status hands out each log line only once, so exactly one poller may call it.
status = CoreStatus()
core_error: str | None = 'not polled yet'
logs: deque[tuple[int, str]] = deque(maxlen=500)
log_seq = itertools.count(1)
_shm: dict[str, shared_memory.SharedMemory] = {}

STATE_COLORS = {'CONNECTED': 'positive', 'CONNECTING': 'warning', 'SETTING_UP': 'warning', 'NEW': 'warning',
                'FAILED': 'negative'}


# Teleop axis bindings copied from the Unity GUI's "Teleoperation" input map.
# (positive key, negative key) for the axes X, Y, Z, RX, RY, RZ; KeyboardEvent.code, so the physical key counts.
KEY_AXES = [('KeyW', 'KeyS'), ('KeyA', 'KeyD'), ('KeyE', 'KeyQ'), ('KeyJ', 'KeyL'), ('KeyK', 'KeyI'), ('KeyO', 'KeyU')]
# Unity used Shift (engage) and Space (gripper); Shift combines with browser shortcuts, so the thumb holds Space
# to engage and G toggles the gripper. Neither has a Firefox/Chrome shortcut on its own.
ENGAGE_KEYS = {'Space'}
GRIPPER_KEY = 'KeyG'
TELEOP_KEYS = {key for pair in KEY_AXES for key in pair} | ENGAGE_KEYS | {GRIPPER_KEY}
STICK_DEADZONE = 0.15  # cheap gamepads rest a little off-centre; without this the arm creeps
TELEOP_CHANNEL = 'teleop_command'  # incar.application.TELEOP_COMMAND_CHANNEL
CONTROLS_HELP = '''
| Motion | Keyboard | Gamepad |
|---|---|---|
| Engage (hold) | Space | A |
| X ± | W / S | left stick up / down |
| Y ± | A / D | left stick left / right |
| Z ± | E / Q | RT / LT |
| Roll ± | J / L | right stick left / right |
| Pitch ± | K / I | right stick down / up |
| Yaw ± | O / U | RB / LB |
| Gripper open/close | G | X |

**Latch engage** keeps the arm engaged without holding a key.
'''


@dataclass
class Teleop:
    """Shared by all tabs: there is one robot and the core accepts one teleop link."""
    keys: set[str] = field(default_factory=set)
    pad: dict | None = None  # latest browser Gamepad snapshot {'axes': [...], 'buttons': [...]}
    pad_time: float = 0.0
    gripper: float = 0.0  # TeleopCommand triggerValue, toggled by G / gamepad X
    hand: str = 'right'
    latched: bool = False  # engaged without holding a key, from the GUI button
    link: WebRTCConnection | None = None


teleop = Teleop()


def teleop_command(keys: set[str], pad: dict | None, gripper: float, hand: str, latched: bool = False) -> TeleopCommand:
    axes = [float((pos in keys) - (neg in keys)) for pos, neg in KEY_AXES]
    engaged = latched or bool(keys & ENGAGE_KEYS)
    if pad:
        lx, ly, rx, ry = (a if abs(a) > STICK_DEADZONE else 0.0 for a in pad['axes'][:4])
        b = pad['buttons']
        # browser "standard" mapping has stick y positive down; signs reproduce Unity's bindings incl. its Invert processors
        pad_axes = [-ly, -lx, b[7] - b[6], -rx, ry, b[5] - b[4]]
        axes = [max(-1.0, min(1.0, k + p)) for k, p in zip(axes, pad_axes)]
        engaged = engaged or b[0] > 0.5
    cmd = TeleopCommand()
    # only the selected hand is set: the core then leaves the other arm and the base (tank) command alone
    v = getattr(cmd, f'{hand}Velocity')
    v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z = axes
    buttons = getattr(cmd, f'{hand}Buttons')
    buttons.primaryButton = engaged
    buttons.gripValue = 1.0 if engaged else 0.0  # core moves an arm only while gripValue > 0.5
    buttons.triggerValue = gripper
    return cmd


def send_teleop() -> None:
    channel = teleop.link and teleop.link.data_channels.get(TELEOP_CHANNEL)
    if not channel or channel.readyState != 'open':
        return
    channel.send(current_teleop_command().SerializeToString())


def current_teleop_command() -> TeleopCommand:
    pad = teleop.pad if time.monotonic() - teleop.pad_time < 0.5 else None  # a frozen tab must not hold the sticks
    return teleop_command(teleop.keys, pad, teleop.gripper, teleop.hand, teleop.latched)


def release_teleop_inputs() -> None:
    teleop.keys.clear()
    teleop.pad = None


def stop_teleop_inputs() -> None:
    """Release everything including the latch: control switched off, tab closed or link dropped."""
    release_teleop_inputs()
    teleop.latched = False


def hmd_waiting() -> bool:
    return (status.HasField('hmdConnectionState')
            and status.hmdConnectionState.state == ConnectionState.CONNECTION_STATE_NEW)


async def open_teleop_link(ip: str, port: int, attempts: int = 40) -> bool:
    """Answer the core's offer on its HMD port, like the Unity GUI and the Quest do.

    The core reports "waiting" a moment before it listens (it builds its offer first), so retry refused connects.
    """
    for _ in range(attempts):
        link = WebRTCConnection()
        background_tasks.create(link.start_connection(ip, port, send_offer=False), name='teleop link')
        await asyncio.sleep(0.25)
        if link.get_status() != 'closed':  # a refused connect closes the link immediately
            teleop.link = link
            return True
    return False


async def open_hmd(port: int) -> bool:
    """Make the core wait for a headset/teleop client on its HMD port; status then shows its address."""
    if not hmd_waiting() and await call('POST', '/hmd/open_connection', params={'port': port}) is None:
        return False
    for _ in range(40):  # the core (re)starts its listener; wait until status shows where
        if hmd_waiting():
            return True
        await asyncio.sleep(0.25)
    ui.notify('The core did not open its HMD connection', type='negative')
    return False


bridge = headset.Bridge()  # pairs XR Commander on the headset with the core, like the INCAR GUI


def headset_qr(payload: str) -> str:
    """The QR code XR Commander scans: the core's compact offer (see headset.compact_offer)."""
    # make_qr, not make: short text would become Micro QR, which ZXing-style scanners can't read
    return segno.make_qr(payload).svg_data_uri(omitsize=True, border=4, light='white')


async def connect_teleop(port: int) -> None:
    if not await open_hmd(port):
        return
    hmd = status.hmdConnectionState
    if not await open_teleop_link(hmd.ip, hmd.port):
        ui.notify(f'Could not reach the core at {hmd.ip}:{hmd.port}', type='negative')


async def disconnect_teleop() -> None:
    link, teleop.link = teleop.link, None
    stop_teleop_inputs()
    if link:
        await link.disconnect()


def teleop_link_state() -> str:
    return teleop.link.get_status() if teleop.link else 'disconnected'


def decode_status(body: dict) -> CoreStatus:
    return CoreStatus.FromString(base64.b64decode(body['status']))


async def poll() -> None:
    global status, core_error
    try:
        r = await client.get('/status')
        r.raise_for_status()
    except httpx.HTTPError as e:
        core_error = f'{type(e).__name__}: {e}'
        return
    status, core_error = decode_status(r.json()), None
    for log in status.logs:
        logs.append((next(log_seq), log.message))


async def call(method: str, path: str, quiet: bool = False, **kwargs) -> dict | None:
    try:
        r = await client.request(method, path, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        if not quiet:
            ui.notify(f'{path}: {e.response.text}', type='negative')
    except httpx.HTTPError as e:
        if not quiet:
            ui.notify(f'{path}: {e}', type='negative')
    return None


def read_frame(name: str, shape) -> np.ndarray | None:
    """Copy the latest frame from incar's shared memory `incar_<name>` (RGB, HxWx3 uint8)."""
    _, h, w = shape
    if name not in _shm:
        try:
            # track=False: never unlink incar's segment when this GUI exits
            _shm[name] = shared_memory.SharedMemory(f'incar_{name}', track=False)
        except FileNotFoundError:
            return None
    shm = _shm[name]
    if shm.size < h * w * 3:
        return None
    return np.ndarray((h, w, 3), np.uint8, shm.buf).copy()


STALE_AFTER = 2.0  # seconds without a changed frame before a feed counts as "no data"
_last_frames: dict[str, tuple[np.ndarray, float]] = {}  # name -> (last frame, time it last changed)


def frame_is_live(name: str, frame: np.ndarray) -> bool:
    """True while the frame keeps changing: the core reports no frame counter or timestamp.
    shortcut: a perfectly static simulated scene also counts as no data; use a frame counter if the core gets one."""
    now = time.monotonic()
    previous = _last_frames.get(name)
    if previous is None or not np.array_equal(previous[0], frame):
        # first sight is not evidence of a live feed: a camera that never delivered still holds its initial frame
        _last_frames[name] = (frame, now - STALE_AFTER if previous is None else now)
        return previous is not None
    return now - previous[1] < STALE_AFTER


def forget_mapping(name: str) -> None:
    """Re-open the segment on the next read: a restarted camera process creates a new one under the same name."""
    _shm.pop(name, None)


def jpeg(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, 'JPEG', quality=70)
    return buf.getvalue()


def camera_live(name: str) -> bool:
    """Whether the camera's frames keep changing, as last seen by its stream (see camera_jpegs)."""
    if host.remote:
        entry = remote_frames.frames.get(name)
        return bool(entry and entry['changed'] and time.monotonic() - entry['changed'] < STALE_AFTER)
    return name in _last_frames and time.monotonic() - _last_frames[name][1] < STALE_AFTER


CAMERA_FPS = 30  # how often a stream looks for a new frame; it only sends changed ones


async def camera_jpegs(name: str):
    """The camera's frames as JPEG, each once, for as long as the core lists the camera."""
    sent = None
    while (stream := next((s for s in status.availableVideoStreams if s.address == name), None)) is not None:
        if host.remote:  # the pump on the remote PC encodes; it only sends changed frames
            entry = remote_frames.frames.get(name)
            if entry and entry['jpeg'] is not sent:
                sent = entry['jpeg']
                yield sent
        elif (frame := await asyncio.to_thread(read_frame, name, stream.shape)) is not None:
            frame_is_live(name, frame)
            if _last_frames[name][1] != sent:  # time of the last change: a new frame
                sent = _last_frames[name][1]
                yield await asyncio.to_thread(jpeg, frame)
        await asyncio.sleep(1 / CAMERA_FPS)


@app.get('/camera/{name}')
async def camera_stream(name: str) -> StreamingResponse:
    """MJPEG: the browser shows each frame as it arrives, without the websocket or a timer in between."""
    if not any(s.address == name for s in status.availableVideoStreams):
        raise HTTPException(404, f'No camera {name}')

    async def parts():
        async for frame in camera_jpegs(name):
            yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%b\r\n' % (len(frame), frame)
    return StreamingResponse(parts(), media_type='multipart/x-mixed-replace; boundary=frame',
                             headers={'Cache-Control': 'no-store'})


def text_segments(texts: list) -> list[tuple[int, int, str]]:
    """Runs of the same text as (first step, last step, text)."""
    segments = []
    for i, text in enumerate(texts):
        if segments and segments[-1][2] == text:
            segments[-1] = (segments[-1][0], i, text)
        else:
            segments.append((i, i, text))
    return segments


def text_colors(texts: list) -> dict:
    """Colour per distinct text in order of first appearance: the timeline segments and the video overlay share it."""
    return {text: training.SERIES_COLORS[n] if n < len(training.SERIES_COLORS) else '#64748b'
            for n, text in enumerate(dict.fromkeys(texts))}


def text_timeline(t: list, texts: list, selection: tuple[int, int]) -> dict:
    """ECharts options: a text feature (e.g. the prompt) as coloured segments over time, one colour per distinct
    text, the selected steps outlined. A step lasts until the next sample (the last one a typical step)."""
    step = ((t[-1] - t[0]) / max(len(t) - 1, 1) if t else 0) or 0.1  # a single sample still gets a width
    end = lambda i: t[i + 1] if i + 1 < len(t) else t[i] + step
    colors = text_colors(texts)
    areas = [[{'xAxis': t[a], 'name': text or '(empty)', 'itemStyle': {'color': colors[text], 'opacity': 0.55}},
              {'xAxis': end(b)}] for a, b, text in text_segments(texts)]
    a, b = selection
    areas.append([{'xAxis': t[a], 'name': '', 'itemStyle': {'color': 'rgba(248,250,252,0.12)', 'borderColor': '#f8fafc',
                                                              'borderWidth': 2}, 'label': {'show': False}},
                  {'xAxis': end(b)}])
    muted = {'color': '#94a3b8'}
    last = round(end(len(t) - 1), 2) if t else 1
    return {
        'backgroundColor': 'transparent',
        'animation': False,
        'grid': {'left': 16, 'right': 16, 'top': 6, 'bottom': 26},
        'tooltip': {'trigger': 'item'},
        'xAxis': {'type': 'value', 'min': 0, 'max': last, 'axisLabel': muted, 'splitLine': {'show': False}},
        'yAxis': {'type': 'value', 'show': False, 'min': 0, 'max': 1},
        # two invisible points: ECharts draws the time cursor (a markLine, see demo_sync.js) only on a series with data
        'series': [{'type': 'line', 'data': [[0, 0], [last, 0]], 'showSymbol': False,
                    'lineStyle': {'width': 0, 'color': 'transparent'},  # not opacity: the cursor would inherit it
                    'markArea': {
            'silent': False, 'data': areas,
            'label': {'position': 'inside', 'color': '#f8fafc', 'fontSize': 12, 'overflow': 'truncate'}}}],
    }


def series_chart(t, data) -> dict:
    """ECharts options: one line per dimension over time, zoomable."""
    t, data = np.asarray(t), np.asarray(data).reshape(len(t), -1)
    muted = {'color': '#94a3b8'}
    return {
        'backgroundColor': 'transparent',
        'animation': False,
        'grid': {'left': 56, 'right': 16, 'top': 34, 'bottom': 56},
        'tooltip': {'trigger': 'axis'},
        'legend': {'type': 'scroll', 'top': 0, 'textStyle': muted},
        'xAxis': {'type': 'value', 'name': 's', 'min': 0, 'max': 'dataMax', 'axisLabel': muted,
                  'splitLine': {'show': False}},
        'yAxis': {'type': 'value', 'scale': True, 'axisLabel': muted,
                  'splitLine': {'lineStyle': {'color': 'rgba(148,163,184,0.12)'}}},
        'dataZoom': [{'type': 'inside'}, {'type': 'slider', 'height': 16, 'bottom': 10}],
        'series': [{'name': f'[{i}]', 'type': 'line', 'showSymbol': False, 'sampling': 'lttb',
                    'data': np.column_stack([t, data[:, i]]).round(4).tolist()} for i in range(data.shape[1])],
    }


MEDIA_CACHE = Path(tempfile.gettempdir()) / 'incar_gui_media'  # videos fetched from a remote incar PC


@app.get('/dataset-media/{dataset}/{demo}/{feature}/data.mp4')
async def dataset_video(dataset: str, demo: str, feature: str) -> FileResponse:
    """Stream a demo video; FileResponse answers range requests, so the player can seek."""
    if any(part in ('', '.', '..') for part in (dataset, demo, feature)):  # no escaping the datasets folder
        raise HTTPException(404)
    try:
        r = await client.get('/workspace/get_current_workspace')
        r.raise_for_status()
    except httpx.HTTPError:
        raise HTTPException(503, 'core unreachable') from None
    workspace = r.json()['current_ws_abs_path']
    remote_path = f'{workspace}/{agent.DATASETS_DIR}/{dataset}/{demo}/{feature}/data.mp4'
    if not host.remote:  # the part check above keeps us in the datasets folder; symlinked datasets are fine
        if not Path(remote_path).is_file():
            raise HTTPException(404)
        return FileResponse(remote_path, media_type='video/mp4')
    # shortcut: fetches the whole video once (base64 over SSH) and caches it; stream in chunks for very long videos
    cached = MEDIA_CACHE / (hashlib.sha1(f'{host.label}:{remote_path}'.encode()).hexdigest() + '.mp4')
    if not cached.is_file():
        try:
            data = base64.b64decode(await host.call('read_file', remote_path))
        except RuntimeError:
            raise HTTPException(404) from None
        MEDIA_CACHE.mkdir(exist_ok=True)
        cached.write_bytes(data)
    return FileResponse(cached, media_type='video/mp4')


_reattach_pending = True


async def watch_training() -> None:
    """Follow the training log; once the core is reachable, re-attach to a training from an earlier GUI run."""
    global _reattach_pending
    if _reattach_pending and core_error is None:
        try:
            r = await client.get('/workspace/get_current_workspace')
            r.raise_for_status()
            await training.reattach(host, r.json()['current_ws_abs_path'])
            _reattach_pending = False
        except (httpx.HTTPError, RuntimeError, OSError, KeyError, ValueError):
            pass  # try again next tick
    await training.poll()


async def watch_frames() -> None:
    """With a remote incar PC, camera frames come from a pump over SSH (shared memory is on that PC)."""
    if host.remote and host.connected and core_error is None:
        streams = [[s.address, s.shape[1], s.shape[2]] for s in status.availableVideoStreams
                   if s.address not in hidden_cameras]
        try:
            await remote_frames.follow(host, streams)
        except OSError:
            pass  # try again next tick


# ─── which PC runs incar ──────────────────────────────────────────────────────────────────────────────────────

SETTINGS_FILE = Path(os.getenv('XDG_CONFIG_HOME') or Path.home() / '.config') / 'incar_gui' / 'connection.json'
DEFAULT_SETTINGS = {'mode': 'local', 'target': '', 'venv': '~/incar', 'core_port': 8000, 'api_key': API_KEY}
host: hosts.LocalHost | hosts.SshHost = hosts.LocalHost()
remote_frames = hosts.RemoteFrames()


def load_settings() -> dict:
    try:
        return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_FILE.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULT_SETTINGS)


hidden_cameras: set[str] = set(load_settings().get('hidden_cameras', []))  # streams this GUI does not show


def toggle_hidden_camera(name: str) -> None:
    hidden_cameras.symmetric_difference_update({name})
    save_settings({**load_settings(), 'hidden_cameras': sorted(hidden_cameras)})


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=4))
    SETTINGS_FILE.chmod(0o600)  # holds the core's API key


async def connect(settings: dict) -> None:
    """Switch to the incar PC described by settings; raises ConnectionError (and keeps the old one) on failure."""
    global host, client, status, core_error, _reattach_pending
    if settings['mode'] == 'ssh':
        new = hosts.SshHost(settings['target'], settings['venv'], int(settings['core_port']))
        await new.connect()
        url = new.core_url
    else:
        new, url = hosts.LocalHost(), CORE_URL
    old_host, old_client = host, client
    host, client = new, httpx.AsyncClient(base_url=url, headers={'X-API-Key': settings['api_key']}, timeout=5)
    status, core_error, _reattach_pending = CoreStatus(), 'connecting', True
    training.runs.clear()  # trainings on the other PC keep running; switching back re-attaches them
    await remote_frames.stop()
    await old_client.aclose()
    if old_host.remote:
        await old_host.close()


async def connect_saved() -> None:
    global core_error
    settings = load_settings()
    if settings['mode'] == 'ssh':
        try:
            await connect(settings)
        except ConnectionError as e:
            core_error = f'SSH to {settings["target"]} failed: {e}'


async def disconnect_remote() -> None:
    await remote_frames.stop()
    if host.remote:
        await host.close()


def connection(s: CoreStatus, attr: str) -> tuple[str, str]:
    """(label, chip color) for robotConnectionState / hmdConnectionState."""
    if not s.HasField(attr):
        return 'not connected', 'grey'
    name = ConnectionState.Name(getattr(s, attr).state).removeprefix('CONNECTION_STATE_')
    return name.lower().replace('_', ' '), STATE_COLORS.get(name, 'grey')


# incoming fields the core records from a TeleopCommand (headset or this GUI's keyboard/gamepad link);
# every other incoming field is robot state or a robot sensor channel
TELEOP_FIELDS = {'left.buttons', 'right.buttons', 'pedals'}


def message_rows(s: CoreStatus) -> tuple[str, dict[str, list[dict]]]:
    """(current prompt, rows per route); incar smuggles the prompt in as a fake field `prompt_<text>`."""
    def row(m) -> dict:
        return {'label': m.label, 'shape': str(list(m.shape)), 'age': f'{-m.deltaTime:.0f} ms'}

    prompt = next((m.label.removeprefix('prompt_') for m in s.incomingMessageFields if m.label.startswith('prompt_')), '')
    incoming = [m for m in s.incomingMessageFields if not m.label.startswith('prompt_')]
    return prompt, {
        'robot_in': [row(m) for m in incoming if m.label not in TELEOP_FIELDS],
        'teleop_in': [row(m) for m in incoming if m.label in TELEOP_FIELDS],
        'robot_out': [row(m) for m in s.outgoingMessageFields],  # commands the core sends to the robot
    }


def live_features(s: CoreStatus) -> dict:
    """What the core receives and sends right now, as dataset features: robot state, commands and cameras."""
    def recordable(m) -> bool:  # teleop buttons and the prompt have no array shape
        return bool(m.shape) and all(m.shape) and m.label not in TELEOP_FIELDS
    return {**{m.label: {'type': 'STATE', 'shape': list(m.shape)} for m in s.incomingMessageFields if recordable(m)},
            **{m.label: {'type': 'ACTION', 'shape': list(m.shape)} for m in s.outgoingMessageFields if recordable(m)},
            **{v.address: {'type': 'VISUAL', 'shape': list(v.shape)} for v in s.availableVideoStreams}}


def index() -> None:
    ui.page_title('incar')
    ui.dark_mode(True)  # console look; theme.css styles the dark variant only

    shown_workspace = None  # the core's workspace as last put into the Settings field

    async def refresh() -> None:
        nonlocal shown_workspace
        # one notice instead of one per list when the core is unreachable (e.g. the remote PC is offline)
        if (r := await call('GET', '/workspace/get_current_workspace', quiet=True)) is None:
            ui.notify(f'The incar core on {host.label} is not reachable; lists not loaded', type='warning')
            return
        if workspace.value in (None, '', shown_workspace):  # never overwrite a path the user is still entering
            workspace.set_value(r['current_ws_abs_path'])
        shown_workspace = r['current_ws_abs_path']
        if (r := await call('GET', '/workspace/dataset/get_available_datasets')) is not None:
            dataset.set_options(r['dataset_folder_names'])
        if (r := await call('GET', '/workspace/policy/get_available_policies')) is not None:
            policy.set_options(r['policy_folder_names'])
        await load_devices()

    async def load_devices() -> None:
        # quiet: a core without input devices (other incar versions answer 404) is no error worth a notice
        r = await call('GET', '/input/available_devices', quiet=True)
        device.set_enabled(r is not None)
        devices_hint.set_text('Input plugins running inside the core, e.g. sinusoid.' if r is not None
                              else 'This incar core offers no input devices.')
        if r is not None:
            device.set_options({'': 'None', **{d: d for d in r['available_devices']}})

    async def set_workspace(path: str | None = None) -> bool:
        """Switch to the workspace (default: the Settings field); for the current one, incar reloads it."""
        if not await call('POST', '/workspace/set_current_workspace', json={'ws_abs_path': path or workspace.value}):
            return False
        await refresh()
        return True

    async def reload_policy(name: str, checkpoint: str) -> bool:
        """Load the policy again, e.g. after its settings changed; not while it runs."""
        return not status.isExecutingPolicy and await call(
            'PUT', '/workspace/policy/select_policy',
            params={'policy_folder_name': name, 'checkpoint': checkpoint}) is not None

    async def edit_policy() -> None:
        if not policy.value:
            ui.notify('Select a policy first', type='warning')
            return
        await policy_editor.open(policy.value, checkpoint.value)

    recording = {'since': None, 'stopping': False}  # when the current recording started (monotonic), auto stop

    async def toggle_recording() -> None:
        if status.isRecording:
            await call('PUT', '/workspace/dataset/stop_recording')
        elif await call('PUT', '/workspace/dataset/start_recording') is not None:
            recording['since'] = time.monotonic()  # more exact than seeing it in the next status

    async def watch_recording() -> None:
        """Stop a recording at the set length (also one started from the headset), then ask whether to keep it."""
        if not status.isRecording:
            if not recording['since'] or time.monotonic() - recording['since'] > 1.5:  # the status lags a click
                recording.update(since=None, stopping=False)
            return
        recording['since'] = recording['since'] or time.monotonic()
        limit = rec_limit.value
        if not limit or recording['stopping'] or time.monotonic() - recording['since'] < limit:
            return
        recording['stopping'] = True
        context = await incar_context()
        if context is None:
            return
        rec_host, ws = context
        folder = f'{ws}/datasets/{status.selectedDataset.name}'
        demos = [n for n in await rec_host.call('list_names', folder) if n.startswith('demo_')]
        demo = max(demos, key=agent._demo_number, default=None)  # incar created it when this recording started
        if await call('PUT', '/workspace/dataset/stop_recording') is not None and demo is not None:  # ask in its own task: this timer must keep watching, e.g. the next recording
            background_tasks.create(ask_keep(rec_host, folder, demo, limit), name='keep recording?',
                                    context=ui.context.client)

    async def ask_keep(rec_host, folder: str, demo: str, limit: float) -> None:
        """Keep or discard the demo an automatic stop produced; closing the question keeps it."""
        if not await training_ui.confirm(f'The recording stopped after {limit:g} s. Keep {demo} in '
                                         f'{folder.rsplit("/", 1)[-1]}?', 'Discard', 'Keep'):
            ui.notify(f'Kept {demo}', type='positive')
            return
        for _ in range(240):  # incar is still writing it (videos): delete only the finished demo
            if await rec_host.call('exists', f'{folder}/{demo}/meta.json'):
                await rec_host.call('remove_demo', folder, demo)
                ui.notify(f'Discarded {demo}', type='positive')
                return
            await asyncio.sleep(0.5)
        ui.notify(f'{demo} was not saved completely; nothing discarded', type='warning')

    def camera_change_blocked() -> str | None:
        return ('incar is recording' if status.isRecording else
                'a policy is running' if status.isExecutingPolicy else None)

    async def load_checkpoints() -> None:
        checkpoint.set_options([], value=None)
        if policy.value and (r := await call('GET', '/workspace/policy/get_available_checkpoints',
                                             params={'policy_folder_name': policy.value})):
            checkpoint.set_options(r['checkpoints'])

    async def send_prompt() -> None:
        await call('PUT', '/sensors/prompt', params={'prompt': prompt.value})

    async def incar_context() -> tuple | None:
        """(host, workspace path on it) of the connected incar PC."""
        r = await call('GET', '/workspace/get_current_workspace')
        return (host, r['current_ws_abs_path']) if r else None

    async def create_dataset(name: str, config: dict) -> bool:
        params = {'dataset_name': name, 'config_string': json.dumps(config, indent=4)}
        if await call('PUT', '/workspace/dataset/create_dataset', params=params) is None:
            return False
        ui.notify(f'Dataset "{name}" created and selected', type='positive')
        await refresh()
        dataset.set_value(name)
        return True

    async def delete_dataset() -> None:
        name = dataset.value
        if not name:
            ui.notify('Select a dataset first', type='warning')
            return
        if status.isRecording and status.selectedDataset.name == name:
            ui.notify('Stop the recording first', type='warning')
            return
        if (r := await call('GET', '/workspace/get_current_workspace')) is None:
            return
        workspace = r['current_ws_abs_path']
        try:
            demos = len(await host.call('list_demos', f'{workspace}/{agent.DATASETS_DIR}/{name}'))
        except (RuntimeError, OSError):
            demos = '?'
        with ui.dialog() as dialog, ui.card().classes('gap-3 p-6 max-w-md'):
            heading('Delete dataset', 'ph-trash')
            ui.label(f'This permanently deletes "{name}" with its {demos} recorded demos on {host.label}. '
                     'Preprocessed copies and policies trained on it stay.')
            typed = ui.input('Type the dataset name to confirm').classes('w-full')
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat')
                ui.button('Delete', icon='ph-trash', on_click=lambda: dialog.submit(True)).props('color=negative') \
                    .bind_enabled_from(typed, 'value', backward=lambda v: v == name)
        confirmed = await dialog
        dialog.delete()
        if not confirmed:
            return
        # the core keeps the selected dataset open: let it go first
        if status.selectedDataset.name == name and await call('PUT', '/workspace/dataset/select_dataset') is None:
            return
        try:
            await host.call('remove_dataset', workspace, name)
        except (RuntimeError, OSError) as e:
            ui.notify(f'Could not delete {name}: {e}', type='negative')
            return
        dataset.set_value(None)
        ui.notify(f'Dataset "{name}" deleted', type='positive')
        await refresh()

    async def open_inspector() -> None:
        name = status.selectedDataset.name or dataset.value
        if not name:
            ui.notify('Select a dataset first', type='warning')
            return
        if (r := await call('GET', '/workspace/get_current_workspace')) is None:
            return
        root = f'{r["current_ws_abs_path"]}/{agent.DATASETS_DIR}/{name}'
        try:
            config = json.loads(await host.call('read_text', f'{root}/dataset_config.json'))
            features, fps = config['features'], config.get('video_fps') or 30
            demos = await host.call('list_demos', root)
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            ui.notify(f'Cannot read {root}: {e}', type='negative')
            return
        for demo in demos:
            demo['start'] = datetime.fromisoformat(demo['start'])
        inspector_title.set_text(name)
        inspector_count.set_text(f'{len(demos)} demos')
        demo_list.clear()
        demo_items.clear()
        with demo_list:
            if not demos:
                ui.label('No demos recorded yet').classes('hint p-3')
            for demo in demos:
                with ui.item(on_click=lambda d=demo: show_demo(root, features, fps, d)).classes('demo-item') as item:
                    with ui.item_section():
                        ui.item_label(demo['name']).classes('font-medium')
                        ui.item_label(f'{demo["start"]:%Y-%m-%d %H:%M:%S} · {demo["duration"]:.1f} s').props('caption')
                    if demo['takeover']:
                        with ui.item_section().props('side'):
                            ui.label('takeover').classes('pill text-xs')
                demo_items[demo['name']] = item
        inspector.open()
        detail.clear()
        if demos:
            await show_demo(root, features, fps, demos[0])

    async def load_feature(path: str, op: str):
        try:
            return await host.call(op, path)
        except (RuntimeError, OSError) as e:
            return e

    unsaved: dict[str, bool] = {}  # text feature folder -> has edits not saved yet
    timeline_clicks: dict[int, Callable] = {}  # text timeline chart id -> what a click at time t does
    ui.on('demo_chart_click', lambda e: timeline_clicks.get(e.args['chart'], lambda t: None)(e.args['t']))

    async def discard_unsaved() -> bool:
        """True if there are no unsaved text edits, or the user lets them go."""
        if any(unsaved.values()) and not await training_ui.confirm('The text edits of this demo are not saved. '
                                                                   'Discard them?', 'Discard'):
            return False
        unsaved.clear()
        inspector.props(remove='persistent')
        return True

    async def close_inspector() -> None:
        if await discard_unsaved():
            inspector.close()

    def text_editor(feature_dir: str, feature: str, values: dict, charts: list) -> None:
        """A text feature (e.g. the prompt) as a timeline of segments; one step or a whole range gets a new text at once."""
        t, texts, saved = values['t'], list(values['text']), list(values['text'])
        if not t:
            ui.label('No samples in this demo').classes('hint')
            return
        last = len(t) - 1
        timeline = ui.echart(text_timeline(t, texts, (0, last)), theme='dark').classes('w-full h-28')
        charts.append(timeline.id)
        steps = ui.range(min=0, max=max(last, 1), value={'min': 0, 'max': last}).classes('w-full px-4')
        selected = ui.label().classes('hint')
        with ui.row().classes('w-full items-center gap-2 no-wrap'):
            new_text = ui.input('New text for the selected steps').classes('grow')
            ui.button('Apply', icon='ph-check', on_click=lambda: apply()).props('flat') \
                .tooltip('Give every selected step this text')
            revert_button = ui.button('Revert', icon='ph-arrow-counter-clockwise', on_click=lambda: revert()).props('flat')
            save_button = ui.button('Save', icon='ph-floppy-disk', on_click=lambda: save())
        table = ui.table(rows=[], columns=[{'name': c, 'label': label, 'field': c, 'align': 'left'} for c, label in
                                           (('step', 'Step'), ('t', 't [s]'), ('value', 'Value'))],
                         row_key='step', pagination=10).classes('w-full').props('flat dense')

        def selection() -> tuple[int, int]:
            v = steps.value or {}
            a, b = sorted((int(v.get('min', 0)), int(v.get('max', 0))))
            return max(0, min(a, last)), max(0, min(b, last))

        def redraw() -> None:
            a, b = selection()
            timeline.options.clear()
            timeline.options.update(text_timeline(t, texts, (a, b)))
            timeline.update()
            shown = set(texts[a:b + 1])
            selected.set_text(f'Selected: step {a}' + (f' to {b} ({b - a + 1} steps)' if b > a else '')
                              + f' · {t[a]:.2f} to {t[b]:.2f} s · '
                              + (f'text "{texts[a]}"' if len(shown) == 1 else f'{len(shown)} different texts'))
            table.rows = [{'step': i, 't': f'{s:.2f}', 'value': v} for i, (s, v) in enumerate(zip(t, texts))]
            table.update()
            dirty = texts != saved
            unsaved[feature_dir] = dirty
            inspector.props('persistent') if any(unsaved.values()) else inspector.props(remove='persistent')
            save_button.set_enabled(dirty)
            revert_button.set_enabled(dirty)
            ui.run_javascript(f'window.demoSetTexts && demoSetTexts({json.dumps(feature)}, {json.dumps(texts)}, '
                              f'{json.dumps(text_colors(texts))})')

        def select(a: int, b: int) -> None:
            steps.set_value({'min': a, 'max': b})  # redraws and seeks through on_value_change

        def apply() -> None:
            a, b = selection()
            texts[a:b + 1] = [new_text.value or ''] * (b - a + 1)
            redraw()

        def revert() -> None:
            texts[:] = saved
            redraw()

        async def save() -> None:
            try:
                await host.call('write_text_series', feature_dir, texts)
            except (RuntimeError, OSError, ValueError) as e:
                ui.notify(f'{feature} not saved: {e}', type='negative', multi_line=True)
                return
            saved[:] = texts
            redraw()
            ui.notify(f'Saved {feature}; the recorded original is kept as data.h5.orig', type='positive')

        def on_range(_) -> None:
            redraw()
            ui.run_javascript(f'window.demoSeek && demoSeek({t[selection()[0]]})')

        def on_click(at: float) -> None:  # a click on the timeline selects the whole segment there
            step = max((i for i, s in enumerate(t) if s <= at), default=0)
            a, b, text = next(seg for seg in text_segments(texts) if seg[0] <= step <= seg[1])
            new_text.set_value(text)
            select(a, b)

        steps.on_value_change(on_range)
        timeline_clicks[timeline.id] = on_click
        table.on('rowClick', lambda e: select(e.args[1]['step'], e.args[1]['step']))
        redraw()

    async def show_demo(root: str, features: dict, fps: float, demo: dict) -> None:
        if not await discard_unsaved():
            return
        timeline_clicks.clear()
        for name, item in demo_items.items():
            item.classes(add='demo-active') if name == demo['name'] else item.classes(remove='demo-active')
        folder = f'{root}/{demo["name"]}'
        videos = [f for f, spec in features.items() if spec.get('type') == 'VISUAL']
        series = [f for f, spec in features.items() if spec.get('type') != 'VISUAL']
        # load everything first (one round trip each over SSH, in parallel), then draw
        loaded = await asyncio.gather(*(load_feature(f'{folder}/{f}/data.mp4', 'exists') for f in videos),
                                      *(load_feature(f'{folder}/{f}', 'frame_times') for f in videos),
                                      *(load_feature(f'{folder}/{f}', 'load_series') for f in series))
        has_video, times, data = loaded[:len(videos)], loaded[len(videos):2 * len(videos)], loaded[2 * len(videos):]
        linked_videos, linked_charts = [], []  # for the time cursor, see static/demo_sync.js
        detail.clear()
        with detail:
            with ui.row().classes('items-center gap-3'):
                ui.label(demo['name']).classes('text-lg font-medium')
                for icon, text in (('ph-calendar-blank', f'{demo["start"]:%Y-%m-%d %H:%M:%S}'),
                                   ('ph-clock', f'{demo["duration"]:.2f} s')):
                    with ui.element('div').classes('pill'):
                        ui.icon(icon)
                        ui.label(text)
                if demo['takeover']:
                    ui.label('takeover').classes('pill')
            if videos:
                ui.label('Video').classes('section-label')
                with ui.element('div').classes('w-full grid grid-cols-1 xl:grid-cols-2 gap-4'):
                    for feature, exists, frame_times in zip(videos, has_video, times):
                        with ui.column().classes('gap-1'):
                            ui.label(feature).classes('hint')
                            if exists is True:
                                with ui.element('div').classes('relative w-full'):
                                    video = ui.video(f'/dataset-media/{root.rsplit("/", 1)[-1]}/{demo["name"]}/{feature}/data.mp4') \
                                        .classes('w-full rounded-xl overflow-hidden')
                                    overlay = ui.element('div').classes('text-overlay')  # the prompt at this frame
                                linked_videos.append({'id': video.id, 'overlay': overlay.id,
                                                      'times': frame_times if isinstance(frame_times, list) else []})
                            else:
                                ui.label('No video file in this demo').classes('hint')
            with ui.row().classes('items-baseline gap-3'):
                ui.label('Data').classes('section-label')
                ui.label('Click a chart to jump there; the white line follows the video.').classes('hint')
            with ui.element('div').classes('w-full grid grid-cols-1 xl:grid-cols-2 gap-4'):
                for feature, values in zip(series, data):
                    with ui.column().classes('gap-1 w-full'):
                        with ui.row().classes('items-center gap-2'):
                            ui.label(feature).classes('font-medium text-sm')
                            ui.label(features[feature].get('type', '')).classes('pill text-xs')
                        if isinstance(values, Exception):
                            ui.label(f'Cannot read: {values}').classes('hint')
                        elif 'text' in values:  # text features (e.g. the prompt): timeline, editable
                            text_editor(f'{folder}/{feature}', feature, values, linked_charts)
                        else:
                            chart = ui.echart(series_chart(values['t'], values['data']), theme='dark')
                            linked_charts.append(chart.classes('w-full h-64').id)
        texts = {f: {'t': v['t'], 'text': v['text'], 'colors': text_colors(v['text'])} for f, v in zip(series, data)
                 if isinstance(v, dict) and 'text' in v}
        ui.run_javascript(f'linkDemo({json.dumps(linked_videos)}, {json.dumps(linked_charts)}, {fps}, {json.dumps(texts)})')

    async def open_for_headset() -> None:
        if not await open_hmd(int(hmd_port.value or 9998)):
            return
        hmd = status.hmdConnectionState
        try:
            payload = await bridge.pair(hmd.ip, hmd.port)
        except (ConnectionError, OSError, ValueError, asyncio.TimeoutError) as e:
            ui.notify(f'Could not get the headset offer from the core: {e}', type='negative')
            return
        qr_image.set_source(headset_qr(payload))
        qr_dialog.open()

    def qr_closed() -> None:
        if not bridge.answered:  # not scanned: free the core's HMD connection, e.g. for keyboard teleop
            bridge.release()

    async def toggle_link() -> None:
        if teleop_link_state() in ('disconnected', 'closed', 'failed'):
            await connect_teleop(int(hmd_port.value or 9998))
        else:
            await disconnect_teleop()

    def set_control(e) -> None:
        keyboard.active = e.value
        if not e.value:
            stop_teleop_inputs()
        ui.run_javascript(f'window.teleopActive = {"true" if e.value else "false"}')

    def on_key(e) -> None:
        if e.action.keydown:
            if e.key.code == GRIPPER_KEY and not e.action.repeat:
                teleop.gripper = 1.0 - teleop.gripper
            teleop.keys.add(e.key.code)
        elif e.action.keyup:
            teleop.keys.discard(e.key.code)

    def on_gamepad(e) -> None:
        if not control.value:
            return
        pressed = e.args['buttons'][2] > 0.5  # X (west) toggles the gripper, like G
        if pressed and not (teleop.pad and teleop.pad['buttons'][2] > 0.5):
            teleop.gripper = 1.0 - teleop.gripper
        teleop.pad, teleop.pad_time = e.args, time.monotonic()

    keyboard = ui.keyboard(on_key=on_key, active=False, ignore=['input', 'textarea'])  # typing never drives the robot
    ui.on('teleop_gamepad', on_gamepad)
    ui.on('teleop_release', release_teleop_inputs)
    ui.context.client.on_disconnect(stop_teleop_inputs)
    ui.add_body_html('''<script>
    setInterval(() => {
      if (!window.teleopActive) return;
      const pad = [...navigator.getGamepads()].find(p => p && p.mapping === 'standard');
      if (pad) emitEvent('teleop_gamepad', {axes: pad.axes.slice(0, 4), buttons: pad.buttons.map(b => b.value)});
    }, 33);
    // keys held while the window loses focus never send keyup
    window.addEventListener('blur', () => { if (window.teleopActive) emitEvent('teleop_release'); });
    // a focused button/switch would swallow or react to Space (engage): drop focus after clicks (capture + deferred,
    // as Quasar stops the click and re-focuses switches); fields keep it, or their menus would close
    document.addEventListener('click', () => setTimeout(() => {
      const el = document.activeElement;
      if (el && el !== document.body && !el.closest('.q-field')) el.blur();
    }), true);
    // teleop keys must not scroll, open a focused dropdown or start find-as-you-type; text inputs still get them
    const teleopKeys = new Set(__TELEOP_KEYS__);
    document.addEventListener('keydown', e => {
      if (window.teleopActive && teleopKeys.has(e.code) && !['INPUT', 'TEXTAREA'].includes(e.target.tagName)) e.preventDefault();
    }, true);
    </script>'''.replace('__TELEOP_KEYS__', json.dumps(sorted(TELEOP_KEYS))))

    with ui.header().classes('glass-bar items-center gap-3 px-6 py-3 flex-wrap'):
        ui.label('L_NK').classes('brand-logo')
        with ui.column().classes('gap-0 brand-credit').props('tabindex=0'):  # hover: who built this (credits.html)
            with ui.row().classes('items-center gap-2 no-wrap'):
                ui.label('incar').classes('brand-name')
                ui.label('alpha').classes('alpha-tag')
            brand_sub = ui.label().classes('brand-sub')
            ui.html((Path(__file__).parent / 'static' / 'credits.html').read_text(), sanitize=False)
        ui.space()
        chips = {key: pill(icon) for key, icon in (('core', 'ph-hard-drives'), ('robot', 'ph-robot'),
                                                   ('hmd', 'ph-virtual-reality'), ('rec', 'ph-record'),
                                                   ('policy', 'ph-cpu'))}
        with chips['core'][0]:
            core_tip = ui.tooltip()
        progress = training_ui.ProgressView(incar_context)
        trainer = training_ui.Trainer(incar_context, progress.open)
        creator = training_ui.DatasetCreator(incar_context, lambda: live_features(status),
                                             lambda: list(dataset.options), create_dataset)
        cameras = training_ui.CameraManager(incar_context, lambda: [s.address for s in status.availableVideoStreams],
                                            hidden_cameras, toggle_hidden_camera, camera_change_blocked, set_workspace)
        policy_editor = training_ui.PolicyEditor(
            incar_context, lambda: (status.selectedPolicy.name, status.selectedPolicy.checkpoint), reload_policy)
        training_ui.header_indicator(progress.open)
        icon_button('ph-arrow-clockwise', 'Refresh lists', refresh)
        icon_button('ph-gear-six', 'Settings', lambda: settings.open())

    def connection_fields() -> dict:
        return {'mode': mode.value, 'target': (target.value or '').strip(), 'venv': (venv.value or '').strip(),
                'core_port': int(core_port.value or 8000), 'api_key': api_key.value}

    def setup_name(fields: dict) -> str:
        return fields['target'] if fields['mode'] == 'ssh' else f'This PC (port {fields["core_port"]})'

    def save_setup() -> None:
        fields = connection_fields()
        if fields['mode'] == 'ssh' and not fields['target']:
            ui.notify('Enter the SSH target first', type='warning')
            return
        stored = load_settings()
        stored['saved'] = {**stored.get('saved', {}), setup_name(fields): fields}  # same target: updated
        save_settings(stored)
        setups.set_options(sorted(stored['saved']), value=setup_name(fields))
        ui.notify(f'Saved setup "{setup_name(fields)}"', type='positive')

    def delete_setup() -> None:
        stored = load_settings()
        if stored.get('saved', {}).pop(setups.value, None) is not None:
            save_settings(stored)
            ui.notify(f'Deleted setup "{setups.value}"', type='positive')
        setups.set_options(sorted(stored.get('saved', {})), value=None)

    async def use_setup() -> None:
        fields = load_settings().get('saved', {}).get(setups.value)
        if not fields or {**DEFAULT_SETTINGS, **fields} == {**DEFAULT_SETTINGS, **connection_fields()}:
            return  # nothing picked, or it is what the fields show already (e.g. right after saving it)
        fields = {**DEFAULT_SETTINGS, **fields}
        for element, key in ((mode, 'mode'), (target, 'target'), (venv, 'venv'), (core_port, 'core_port'),
                             (api_key, 'api_key')):
            element.set_value(fields[key])
        await apply_connection()

    async def apply_connection() -> None:
        settings_now = connection_fields()
        if settings_now['mode'] == 'ssh' and not settings_now['target']:
            ui.notify('Enter the SSH target, e.g. user@robot-pc', type='warning')
            return
        connect_button.props('loading')
        connection_state.set_text('Connecting…')
        try:
            await connect(settings_now)
        except ConnectionError as e:
            connection_state.set_text(f'Not connected: {e}')
            ui.notify(f'Could not connect: {e}', type='negative', multi_line=True)
            return
        finally:
            connect_button.props(remove='loading')
        save_settings({**load_settings(), **settings_now})  # keeps the hidden cameras
        connection_state.set_text(f'Connected to {host.label}')
        ui.notify(f'Using incar on {host.label}', type='positive')
        await refresh()

    saved = load_settings()
    with ui.dialog() as settings, ui.card().classes('w-full max-w-xl gap-3'):
        heading('Settings', 'ph-gear-six')
        ui.label('incar PC').classes('section-label')
        with ui.row().classes('w-full items-center no-wrap gap-2'):
            setups = ui.select(sorted(saved.get('saved', {})), label='Saved setups', on_change=use_setup) \
                .classes('grow').props('hint="Pick one to fill in its connection and connect"')
            icon_button('ph-trash', 'Delete this saved setup', delete_setup)
        mode = ui.toggle({'local': 'This PC', 'ssh': 'Remote PC over SSH'}, value=saved['mode']).props('no-caps')
        with ui.column().classes('w-full gap-3') as ssh_fields:
            target = ui.input('SSH target', value=saved['target'],
                              placeholder='user@robot-pc, or a Host from ~/.ssh/config').classes('w-full')
            with ui.row().classes('w-full no-wrap gap-3'):
                venv = ui.input('incar environment on that PC', value=saved['venv']).classes('grow') \
                    .props('hint="The venv folder the incar installer created"')
                core_port = ui.number('Core port', value=saved['core_port'], precision=0).classes('w-28')
            ui.label('Uses your ssh setup (~/.ssh/config, keys, agent). Logging in must work without a password '
                     'prompt (e.g. after ssh-copy-id) and the host must be in known_hosts (ssh to it once). '
                     'Keyboard/gamepad teleop and the headset connect to that PC directly, so they need network '
                     'access to it (same LAN or VPN), not just SSH.').classes('hint')
        ssh_fields.bind_visibility_from(mode, 'value', value='ssh')
        api_key = ui.input('Core API key', value=saved['api_key'], password=True, password_toggle_button=True) \
            .classes('w-full')
        with ui.row().classes('w-full items-center gap-3'):
            connect_button = ui.button('Connect', icon='ph-plug', on_click=apply_connection)
            ui.button('Save', icon='ph-bookmark-simple', on_click=save_setup).props('flat') \
                .tooltip('Keep this connection under "Saved setups"')
            connection_state = ui.label().classes('hint')
        ui.separator()
        ui.label('Workspace').classes('section-label')
        workspace = ui.input('Workspace path').classes('w-full').on('keydown.enter', lambda: set_workspace())
        with workspace.add_slot('append'):
            icon_button('ph-check', 'Set workspace', lambda: set_workspace())
        ui.label('Folder with datasets, policies and workspace_config.json, on the incar PC. '
                 'Press Enter or ✓ to switch to it.').classes('hint')
        with ui.row().classes('w-full justify-end'):
            ui.button('Close', on_click=settings.close).props('flat')

    with ui.dialog().props('maximized') as inspector, ui.card().classes('w-full h-full gap-4 no-wrap p-5'):
        with ui.row().classes('w-full items-center gap-3 no-wrap'):
            heading('Dataset', 'ph-film-strip')
            inspector_title = ui.label().classes('text-lg font-medium')
            inspector_count = ui.label().classes('hint')
            ui.space()
            icon_button('ph-x', 'Close', close_inspector)
        with ui.row().classes('w-full grow no-wrap gap-4 min-h-0'):
            demo_list = ui.list().classes('w-72 shrink-0 h-full overflow-y-auto glass p-2')
            detail = ui.column().classes('grow h-full overflow-y-auto gap-3 pr-2 min-w-0')
    demo_items: dict[str, ui.item] = {}

    with ui.dialog().props('maximized').on('hide', qr_closed) as qr_dialog, \
            ui.card().classes('w-full h-full items-center justify-center gap-3 p-6 no-wrap'):
        heading('Scan with the headset', 'ph-virtual-reality')
        # as large as the screen allows: the headset camera reads a dense code more easily
        qr_image = ui.image().classes('rounded-xl shrink-0').style('width: min(72vh, 92vw); height: min(72vh, 92vw)') \
            .props('fit=contain no-transition alt="Headset connection QR code"')
        ui.label('On the VR headset, open').classes('text-center')
        ui.link(headset.COMMANDER_ORIGIN.removeprefix('https://'), headset.COMMANDER_ORIGIN, new_tab=True) \
            .classes('font-mono text-lg')
        ui.label('select the XR Commander and scan this code. It closes when the headset connects. '
                 f'The headset must reach this PC on port {headset.BRIDGE_PORT}.').classes('hint text-center max-w-80')
        ui.button('Close', on_click=qr_dialog.close).props('flat')

    with ui.element('div').classes('w-full grid grid-cols-1 lg:grid-cols-4 gap-4 items-start'):
        with ui.column().classes('gap-4'):
            with card('Teleop', 'ph-game-controller') as teleop_card:
                with ui.tabs().props('dense no-caps align=justify').classes('w-full') as teleop_tabs:  # fits the card
                    ui.tab('headset', 'Headset', icon='ph-virtual-reality')
                    ui.tab('keys', 'Keys & gamepad', icon='ph-keyboard')
                    ui.tab('core', 'Core input', icon='ph-cpu')
                with ui.tab_panels(teleop_tabs, value='headset').classes('w-full'):
                    with ui.tab_panel('headset').classes('p-0 flex flex-col gap-3'):
                        with ui.row().classes('w-full no-wrap items-center'):
                            hmd_port = ui.number('HMD port', value=9998, format='%d').classes('w-28')
                            ui.button('Open for headset', icon='ph-virtual-reality', on_click=open_for_headset).props('flat')
                        ui.label('The headset and keyboard/gamepad share this connection.').classes('hint')

                    with ui.tab_panel('keys').classes('p-0 flex flex-col gap-3'):
                        control = ui.switch('Control from this tab', on_change=set_control)
                        ui.toggle({'left': 'Left arm', 'right': 'Right arm'}).bind_value(teleop, 'hand').props('no-caps')
                        with ui.row().classes('items-center gap-2'):
                            link_button = ui.button(on_click=toggle_link)
                            link_state = ui.label().classes('hint')
                        latch_button = ui.button(on_click=lambda: setattr(teleop, 'latched', not teleop.latched))
                        with ui.row().classes('gap-2'):
                            engaged_pill = pill('ph-power')
                            gripper_pill = pill('ph-arrows-in-line-vertical')
                        axis_bars = []
                        for name in ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'):
                            with ui.element('div').classes('axis'):
                                ui.label(name)
                                with ui.element('div').classes('axis-track'):
                                    fill = ui.element('div').classes('axis-fill')
                                axis_bars.append((fill, ui.label()))
                        with ui.expansion('Controls', icon='ph-keyboard').classes('w-full'):
                            ui.markdown(CONTROLS_HELP)

                    with ui.tab_panel('core').classes('p-0 flex flex-col gap-3'):
                        # '' tells the core to disable its input device; the core doesn't report the active one
                        device = ui.select({'': 'None'}, label='Core input device', on_change=lambda e: e.value is not None
                                           and call('PUT', '/input/select_device', params={'device': e.value})) \
                            .classes('w-full')
                        device.on('popup-show', load_devices)  # re-read the list every time it opens
                        devices_hint = ui.label('Input plugins running inside the core, e.g. sinusoid.').classes('hint')
                        prompt = ui.input('Prompt').classes('w-full').on('keydown.enter', send_prompt)
                        with prompt.add_slot('append'):
                            icon_button('ph-paper-plane-tilt', 'Send prompt', send_prompt)
                        current_prompt = ui.label().classes('hint')

        with ui.column().classes('lg:col-span-2 gap-4'):
            with card('Cameras', 'ph-video-camera'):
                icon_button('ph-sliders-horizontal', 'Manage cameras', cameras.open).classes('absolute top-3 right-3')
                with ui.element('div').classes('cam-empty w-full') as no_video:
                    ui.icon('ph-video-camera-slash', size='md')
                    no_video_text = ui.label()
                videos = ui.element('div').classes('w-full grid grid-cols-1 md:grid-cols-2 gap-3')
            with ui.card().classes('w-full glass').props('flat'):
                with ui.tabs().props('dense no-caps inline-label align=left') as tabs:
                    ui.tab('messages', 'Messages', icon='ph-arrows-left-right')
                    ui.tab('log', 'Log', icon='ph-scroll')
                with ui.tab_panels(tabs, value='messages').classes('w-full'):
                    with ui.tab_panel('messages').classes('p-0'):
                        columns = [{'name': c, 'label': c.capitalize(), 'field': c, 'align': 'left'}
                                   for c in ('label', 'shape', 'age')]
                        robot, core, teleop_src = ('ph-robot', 'Robot'), ('ph-hard-drives', 'Core'), \
                            ('ph-game-controller', 'Teleop (headset / keyboard / gamepad)')
                        tables = {}
                        with ui.element('div').classes('w-full grid grid-cols-1 md:grid-cols-2 gap-4'):
                            for title, routes in (('Incoming', (('robot_in', robot, core), ('teleop_in', teleop_src, core))),
                                                  ('Outgoing', (('robot_out', core, robot),))):
                                with ui.column().classes('gap-2'):
                                    ui.label(title).classes('section-label')
                                    for key, (src_icon, src), (dst_icon, dst) in routes:
                                        with ui.row().classes('items-center gap-1 text-sm font-medium no-wrap'):
                                            ui.icon(src_icon, color='accent')
                                            ui.label(src)
                                            ui.icon('ph-arrow-right', color='grey').classes('mx-1')
                                            ui.icon(dst_icon, color='accent')
                                            ui.label(dst)
                                        tables[key] = ui.table(rows=[], columns=columns, row_key='label') \
                                            .classes('w-full').props('flat dense')
                    with ui.tab_panel('log').classes('p-0'):
                        log_view = ui.log(max_lines=500).classes('w-full h-56')

        with ui.column().classes('gap-4'):
            with card('Robot', 'ph-robot') as robot_card:
                with ui.row().classes('w-full no-wrap'):
                    ip = ui.input('Robot IP', value='127.0.0.1').classes('grow')
                    port = ui.number('Port', value=9999, format='%d').classes('w-24')
                with ui.row():
                    ui.button('Connect', icon='ph-plug', on_click=lambda: call(
                        'POST', '/robot/connect', params={'ip': ip.value, 'port': int(port.value or 0)}))
                    ui.button('Disconnect', icon='ph-x-circle', on_click=lambda: call(
                        'POST', '/robot/disconnect')).props('flat')

            with card('Dataset', 'ph-database'):
                dataset = ui.select([], label='Dataset', on_change=lambda e: e.value and call(
                    'PUT', '/workspace/dataset/select_dataset', params={'dataset_folder_name': e.value})
                ).classes('w-full')
                demos = ui.label().classes('hint')
                with ui.row().classes('gap-1'):
                    ui.button('Inspect', icon='ph-film-strip', on_click=open_inspector).props('flat')
                    ui.button('New dataset', icon='ph-folder-simple-plus', on_click=creator.open).props('flat')
                    ui.button('Delete', icon='ph-trash', on_click=delete_dataset).props('flat color=negative')
                with ui.row().classes('items-center gap-2'):  # camera-style pair: record/stop and cancel the take
                    rec_button = ui.button(on_click=toggle_recording).classes('rec-control')
                    discard_button = ui.button('Discard', icon='ph-x', on_click=lambda: call(
                        'PUT', '/workspace/dataset/discard_recording')).props('outline color=negative') \
                        .classes('rec-control').tooltip('Throw away the recording in progress')
                rec_limit = ui.number('Stop after (s)', min=1, format='%.0f', placeholder='no limit') \
                    .props('clearable hint="Stops the recording and asks whether to keep it"').classes('w-full')
                ui.select({0: 'Record takeover only', 1: 'Record everything'}, label='During rollout',
                          on_change=lambda e: call('PUT', '/workspace/dataset/rollout_recording_mode',
                                                   params={'mode': e.value})).classes('w-full')

            with card('Policy', 'ph-cpu'):
                policy = ui.select([], label='Policy', on_change=load_checkpoints).classes('w-full')
                checkpoint = ui.select([], label='Checkpoint').classes('w-full')
                loaded = ui.label().classes('hint')
                with ui.row():
                    ui.button('Load', icon='ph-download-simple', on_click=lambda: call(
                        'PUT', '/workspace/policy/select_policy',
                        params={'policy_folder_name': policy.value, 'checkpoint': checkpoint.value})).props('flat')
                    run_button = ui.button(on_click=lambda: call('POST', '/workspace/policy/toggle_policy_execution'))
                ui.button('Settings', icon='ph-sliders-horizontal', on_click=edit_policy).props('flat') \
                    .tooltip('See and edit the settings of the selected policy checkpoint')
                ui.separator()
                with ui.row():
                    ui.button('Train new policy', icon='ph-graduation-cap', on_click=trainer.open).props('flat')
                    ui.button('Trainings', icon='ph-clock-counter-clockwise', on_click=lambda: progress.open()).props('flat') \
                        .tooltip('Progress of running trainings and results of past ones')

    tiles: dict[str, list] = {}  # stream -> [image, badge, overlay, last remote jpeg shown, tile]
    seen = 0
    lists_loaded_for = None  # the host whose datasets/policies are in the dropdowns

    def chip(key: str, text: str, color: str) -> None:
        state = 'rec' if key == 'rec' and color == 'negative' else PILL_STATES[color]  # recording pulses
        set_pill(*chips[key], text, state)

    def render() -> None:
        nonlocal seen, lists_loaded_for
        s = status
        brand_sub.set_text('web GUI' if not host.remote else f'web GUI · {host.label}')
        # reload the lists when the core becomes reachable or the incar PC changes, or an open page stays empty
        if core_error is not None:
            lists_loaded_for = None  # reload once the core is back, e.g. after a core restart
        elif lists_loaded_for is not host:
            lists_loaded_for = host
            ui.timer(0.01, refresh, once=True)
        chip('core', 'core online' if core_error is None else 'core offline',
             'positive' if core_error is None else 'negative')
        core_tip.set_text(core_error or CORE_URL)
        for key, attr in (('robot', 'robotConnectionState'), ('hmd', 'hmdConnectionState')):
            text, color = connection(s, attr)
            chip(key, f'{key} {text}', color)
            # the whole panel shows the connection: green, amber while connecting, red on failure; the headset
            # connection is the teleop one (keyboard/gamepad use it too)
            set_props(robot_card if key == 'robot' else teleop_card, f'data-state={PILL_STATES[color]}')
        if qr_dialog.value and connection(s, 'hmdConnectionState')[0] == 'connected':
            qr_dialog.close()
            ui.notify('Headset connected', type='positive')
        chip('rec', 'recording' if s.isRecording else 'not recording', 'negative' if s.isRecording else 'grey')
        chip('policy', 'policy running' if s.isExecutingPolicy else 'policy idle',
             'positive' if s.isExecutingPolicy else 'grey')

        left = rec_limit.value - (time.monotonic() - recording['since']) \
            if s.isRecording and rec_limit.value and recording['since'] else None
        rec_button.set_text(('Stop' + (f' · {max(left, 0):.0f} s' if left is not None else '')) if s.isRecording
                            else 'Record')
        set_props(rec_button, f'icon={"ph-stop" if s.isRecording else "ph-record"} '
                              f'color={"warning" if s.isRecording else "negative"}')
        if discard_button.enabled != s.isRecording:  # only a recording in progress can be discarded
            discard_button.set_enabled(s.isRecording)
        run_button.set_text('Stop policy' if s.isExecutingPolicy else 'Run policy')
        set_props(run_button, f'icon={"ph-stop" if s.isExecutingPolicy else "ph-play"} '
                         f'color={"negative" if s.isExecutingPolicy else "positive"}')
        demos.set_text(f'{s.selectedDataset.numDemos} demos recorded' if s.selectedDataset.name else 'No dataset selected')
        loaded.set_text(f'Loaded: {s.selectedPolicy.name} @ {s.selectedPolicy.checkpoint}'
                        if s.selectedPolicy.name else 'No policy loaded')

        link = teleop_link_state()
        linked = link not in ('disconnected', 'closed', 'failed')
        link_button.set_text('Disconnect teleop' if linked else 'Connect teleop')
        set_props(link_button, f'icon={"ph-x-circle" if linked else "ph-plug"} color={"negative" if linked else "primary"}')
        link_state.set_text(f'link: {link}')
        latch_button.set_text('Engaged (click to release)' if teleop.latched else 'Latch engage')
        set_props(latch_button, f'icon={"ph-lock" if teleop.latched else "ph-lock-open"} '
                           f'color={"negative" if teleop.latched else "primary"}')

        if covered():
            return  # tables and log catch up once the full-screen window closes
        prompt_text, routes = message_rows(s)
        current_prompt.set_text(f'Current prompt: {prompt_text or "-"}')
        for key, rows in routes.items():
            if tables[key].rows != rows:  # update_rows always pushes the whole table
                tables[key].update_rows(rows)

        for seq, line in logs:
            if seq > seen:
                log_view.push(line)
                seen = seq

    def covered() -> bool:
        """A full-screen window (trainer, progress, dataset inspector or creator, policy settings) hides the page."""
        return bool(trainer.dialog.value or progress.dialog.value or inspector.value or creator.dialog.value
                    or policy_editor.dialog.value)

    def render_teleop() -> None:
        if covered():
            return
        cmd = current_teleop_command()
        v, b = getattr(cmd, f'{teleop.hand}Velocity'), getattr(cmd, f'{teleop.hand}Buttons')
        engaged = b.gripValue > 0.5
        set_pill(*engaged_pill, 'Engaged' if engaged else 'Not engaged', 'ok' if engaged else 'idle')
        set_pill(*gripper_pill, 'Gripper closed' if b.triggerValue else 'Gripper open', 'warn' if b.triggerValue else 'idle')
        axes = (v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z)
        for (fill, value), a in zip(axis_bars, axes):
            set_style(fill, f'left: {50 + min(a, 0) * 50}%; width: {abs(a) * 50}%')
            value.set_text(f'{a:+.2f}')

    def render_video() -> None:
        """Tiles and their "no data" state; the frames themselves come from /camera/<name> (MJPEG)."""
        all_streams = list(status.availableVideoStreams)
        streams = [s for s in all_streams if s.address not in hidden_cameras]
        for name in set(tiles) - {s.address for s in streams}:  # hidden, or removed from the workspace
            tiles.pop(name)[4].delete()
        for stream in streams:
            name, (_, h, w) = stream.address, stream.shape
            if name not in tiles:
                with videos, ui.element('div').classes('cam-tile').style(f'aspect-ratio: {w} / {h}') as tile:
                    image = ui.element('img').props(f'alt="{name}"')
                    with ui.element('div').classes('pill cam-badge') as badge:
                        ui.element('span').classes('dot')
                        ui.label(name)
                    with ui.element('div').classes('cam-nodata') as overlay:
                        ui.icon('ph-video-camera-slash')
                        ui.label('No data received')
                tiles[name] = [image, badge, overlay, None, tile]  # [3]: unused
        no_video.set_visibility(not tiles)
        no_video_text.set_text(f'{len(all_streams)} hidden camera{"s" * (len(all_streams) != 1)}' if all_streams
                               else 'No camera streams yet')
        hidden_page = covered()  # a full-screen window: stop the streams instead of decoding frames nobody sees
        for stream in streams:
            name = stream.address
            image, badge, overlay, _, _ = tiles[name]
            live = camera_live(name)
            if not live and not host.remote:
                forget_mapping(name)  # re-open the segment: a restarted camera process creates a new one
            # a dead feed reconnects every 5 s: the stream may have ended with a core or SSH restart
            source = '' if hidden_page else f'/camera/{urllib.parse.quote(name)}' + ('' if live else f'?{int(time.time()) // 5}')
            set_props(image, f'src="{source}"')
            overlay.set_visibility(not live)
            set_props(badge, f'data-state={"ok" if live else "bad"}')

    render()  # fill chips and button labels before the first tick
    render_teleop()
    ui.timer(0.5, render)
    ui.timer(0.1, watch_recording)
    ui.timer(0.1, render_teleop)  # the axis bars should follow the keys without visible lag
    ui.timer(0.5, render_video)


if __name__ == '__main__':
    for element in (ui.input, ui.number, ui.select):
        element.default_props('outlined dense')
    ui.button.default_props('unelevated no-caps')
    app.colors(primary='#4f6bed', accent='#38bdf8', positive='#16a34a', negative='#e11d48', warning='#d97706')
    static = Path(__file__).parent / 'static'
    app.add_static_files('/static', static)
    # static files are cached for an hour: version our own files by mtime so changes reach open browsers
    version = int(max((static / name).stat().st_mtime for name in ('theme.css', 'demo_sync.js', 'phosphor/icons.js')))
    ui.add_head_html('<link rel="stylesheet" href="/static/phosphor/duotone.css">'
                     f'<script src="/static/phosphor/icons.js?v={version}"></script>'
                     f'<link rel="stylesheet" href="/static/theme.css?v={version}">'
                     f'<script src="/static/demo_sync.js?v={version}"></script>', shared=True)
    app.timer(0.5, poll)
    app.timer(1 / 30, send_teleop)  # the core drops teleop commands older than 0.2 s
    app.timer(1.0, watch_training)
    app.timer(1.0, watch_frames)
    app.on_startup(connect_saved)  # reconnect to the remote incar PC used last time
    app.on_shutdown(disconnect_remote)
    app.on_shutdown(bridge.stop)
    ui.run(index, host=os.getenv('GUI_HOST', '127.0.0.1'), port=int(os.getenv('GUI_PORT', '8090')),
           title='incar', show=False, reload=False)
