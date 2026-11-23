"""Self-check: `.venv/bin/python test_app.py`"""
import asyncio
import base64
import json
import sys
import tempfile
import urllib.parse
from pathlib import Path

import h5py
import httpx
from multiprocessing import shared_memory

import numpy as np

import agent
import app
import headset
import hosts
from incar_networking.messages.core_status_pb2 import CoreStatus, MessageInfo

# status round trip, the way incar's /status encodes it
s = CoreStatus(isRecording=True, incomingMessageFields=[MessageInfo(label='prompt_pick up', shape=[]),
                                                         MessageInfo(label='left.arm', shape=[7], deltaTime=-12)])
s.robotConnectionState.state = 1  # CONNECTED
decoded = app.decode_status({'status': base64.b64encode(s.SerializeToString()).decode()})
assert decoded.isRecording
decoded.outgoingMessageFields.add(label='left.cmd', shape=[6], deltaTime=-3)
decoded.incomingMessageFields.add(label='right.buttons', shape=[0], deltaTime=-5)
prompt, routes = app.message_rows(decoded)
assert prompt == 'pick up', prompt
decoded.availableVideoStreams.add(address='cam', shape=[3, 4, 6])
assert app.live_features(decoded) == {'left.arm': {'type': 'STATE', 'shape': [7]},
                                      'left.cmd': {'type': 'ACTION', 'shape': [6]},
                                      'cam': {'type': 'VISUAL', 'shape': [3, 4, 6]}}, 'no prompt, no teleop buttons'
assert routes == {'robot_in': [{'label': 'left.arm', 'shape': '[7]', 'age': '12 ms'}],
                  'teleop_in': [{'label': 'right.buttons', 'shape': '[0]', 'age': '5 ms'}],
                  'robot_out': [{'label': 'left.cmd', 'shape': '[6]', 'age': '3 ms'}]}, routes
assert app.connection(decoded, 'robotConnectionState') == ('connected', 'positive')
assert app.connection(decoded, 'hmdConnectionState') == ('not connected', 'grey')

# shared-memory frame, written the way incar's camera process does
h, w = 4, 6
seg = shared_memory.SharedMemory('incar_selftest', create=True, size=h * w * 3)
try:
    np.ndarray((h, w, 3), np.uint8, seg.buf)[:] = 7
    frame = app.read_frame('selftest', [3, h, w])
    assert frame.shape == (h, w, 3) and (frame == 7).all()
    assert app.read_frame('missing_cam', [3, h, w]) is None
    assert app.jpeg(frame).startswith(b'\xff\xd8'), 'JPEG start of image'
finally:
    seg.close()
    seg.unlink()

# camera liveness: a feed is live only while its frames change
still, moving = np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)
assert not app.frame_is_live('cam', still), 'first sight is no evidence of data'
assert not app.frame_is_live('cam', still), 'a never-changing frame is no data'
assert app.frame_is_live('cam', moving), 'a changed frame is live'
assert app.frame_is_live('cam', moving), 'still live right after a change'
app._last_frames['cam'] = (moving, app._last_frames['cam'][1] - app.STALE_AFTER)  # age the last change
assert not app.frame_is_live('cam', moving), 'unchanged for STALE_AFTER -> no data'

# dataset inspector: incar's on-disk layout (datasets/<name>/demo_N/{meta.json, <feature>/data.h5|data.mp4})
with tempfile.TemporaryDirectory() as ws:
    ds = Path(ws) / 'datasets' / 'cups'
    for n, takeover in ((10, True), (2, False)):
        (ds / f'demo_{n}' / 'arm').mkdir(parents=True)
        (ds / f'demo_{n}' / 'meta.json').write_text(json.dumps(
            {'t_start': '2026-11-09T09:32:14.000000', 't_end': '2026-11-09T09:32:24.500000', 'is_takeover': takeover}))
        with h5py.File(ds / f'demo_{n}' / 'arm' / 'data.h5', 'w') as f:
            f['data'] = np.arange(6, dtype=np.float32).reshape(3, 2)
            f['timestamps'] = np.array([0, 500_000, 1_000_000])
    (ds / 'demo_3').mkdir()  # still being saved: no meta.json yet
    (ds / 'demo_2' / 'cam').mkdir()
    (ds / 'demo_2' / 'cam' / 'data.mp4').write_bytes(b'\0' * 1000)
    demos = agent.list_demos(str(ds))
    assert [(d['name'], d['duration'], d['takeover']) for d in demos] == [('demo_2', 10.5, False), ('demo_10', 10.5, True)], demos
    with h5py.File(ds / 'demo_2' / 'cam' / 'timestamps.h5', 'w') as f:
        f['timestamps'] = np.array([20_867, 57_845])
    assert agent.frame_times(str(ds / 'demo_2' / 'cam')) == [0.0209, 0.0578]
    assert agent.frame_times(str(ds / 'demo_10' / 'cam')) == [], 'missing timestamps -> plain video time'
    series = agent.load_series(str(ds / 'demo_2' / 'arm'))
    assert series == {'t': [0, 0.5, 1.0], 'data': [[0, 1], [2, 3], [4, 5]]}, series
    json.dumps(series)  # crosses SSH as JSON
    chart = app.series_chart(series['t'], series['data'])
    assert [s['name'] for s in chart['series']] == ['[0]', '[1]'] and chart['series'][1]['data'] == [[0, 1], [0.5, 3], [1, 5]]

    # video route: serves byte ranges (seeking) and never leaves the datasets folder
    from fastapi.testclient import TestClient
    core = httpx.AsyncClient(base_url='http://core', transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={'current_ws_abs_path': ws})))
    app.client, real_client = core, app.client
    (Path(ws) / 'secret' / 'x' / 'y').mkdir(parents=True)
    (Path(ws) / 'secret' / 'x' / 'y' / 'data.mp4').write_bytes(b'private')
    try:
        web = TestClient(app.app)
        r = web.get('/dataset-media/cups/demo_2/cam/data.mp4', headers={'Range': 'bytes=0-99'})
        assert r.status_code == 206 and len(r.content) == 100, (r.status_code, len(r.content))
        assert web.get('/dataset-media/%2E%2E/secret/x/data.mp4').status_code == 404, 'path traversal'
        assert web.get('/dataset-media/cups/demo_9/cam/data.mp4').status_code == 404
    finally:
        app.client = real_client

# deleting datasets: only real dataset folders in datasets/, a symlinked dataset only loses its link
with tempfile.TemporaryDirectory() as ws:
    shared = Path(ws) / 'shared_cups'
    for folder in (Path(ws) / 'datasets' / 'cups', shared):
        folder.mkdir(parents=True)
        (folder / 'dataset_config.json').write_text('{}')
    (Path(ws) / 'datasets' / 'linked').symlink_to(shared)
    (Path(ws) / 'datasets' / 'notes').mkdir()
    for bad in ('..', 'a/b', 'notes', 'missing'):
        try:
            agent.remove_dataset(ws, bad)
            raise AssertionError(f'{bad} must be refused')
        except (ValueError, FileNotFoundError):
            pass
    agent.remove_dataset(ws, 'cups')
    agent.remove_dataset(ws, 'linked')
    assert agent.list_names(f'{ws}/datasets') == ['notes'] and (shared / 'dataset_config.json').is_file()

# training: progress parsed from incar's log format ("LEVEL date time <15 chars file:line> message")
import training


def log(msg: str, level: str = 'INFO', t: str = '2026-11-15 17:00:00') -> str:
    return f'{level} {t} {"_train.py:361":>15} {msg}'


p = training.Progress(total=1000, log_freq=200, validate_freq=400)
assert p.feed(log('Creating datasets')) == 'phase' and p.phase == 'Preparing data'
assert p.feed('Sample dt:  71%|███████▏  | 25/35 [09:14<04:33, 27.39s/it]') == 'task'
assert p.task == {'name': 'Sample dt', 'done': 25, 'total': 35, 'left': '04:33'}, p.task
p.feed(log('Start offline training on a fixed dataset'))
assert p.task is None, 'the preprocessing bar ends when training starts'
assert p.feed(log('step:200 smpl:13K ep:40 epch:0.50 loss:0.500 grdn:1.2 lr:1.0e-04', t='2026-11-15 17:00:00')) == 'step'
p.feed(log('step:400 smpl:26K ep:80 epch:1.00 loss:0.250 grdn:1.1 lr:1.0e-04', t='2026-11-15 17:00:10'))
assert p.step == 400 and p.loss == [[200, 0.5], [400, 0.25]] and p.rate == 20.0 and p.eta() == 30.0
p.feed(log('Ran validation with an action mse of 0.125'))  # printed before the step-800 line
p.feed(log('Saving new best model at step 800'))
p.feed(log('step:1K smpl:51K ep:160 epch:2.00 loss:0.100 grdn:1.0 lr:1.0e-04', t='2026-11-15 17:00:20'))
assert p.val == [[800, 0.125]] and p.best == (800, 0.125) and p.step == 800, (p.val, p.best, p.step)
p.feed(log('Checkpoint policy after step 1000'))
assert p.checkpoints == [1000] and p.step == 1000
assert p.feed('ValueError: shapes do not match') == 'error' and p.error == 'ValueError: shapes do not match'
p.feed(log('End of training'))
assert p.phase == 'Finished'
chart = training.loss_chart(p)
assert chart['yAxis']['type'] == 'log' and chart['series'][1]['markPoint']['data'][0]['coord'] == [800, 0.125]

schema_stub = {'registries': {'step': {'sample_dt': [{'name': 'hooks', 'default': ['DATASET']}],
                                       'limit_velocity': [{'name': 'hooks', 'default': ['TELEOP_COMMAND']}]}}}
assert training.training_step_types(schema_stub) == ['sample_dt'], 'command-only steps are not preprocessing'

with tempfile.TemporaryDirectory() as tmp:
    ws = Path(tmp)
    (ws / 'datasets' / 'cups').mkdir(parents=True)
    (ws / 'datasets' / 'cups' / 'dataset_config.json').write_text(json.dumps({'features': {
        'cam': {'type': 'VISUAL', 'shape': [3, 4, 4]}, 'joints': {'type': 'STATE', 'shape': [6]},
        'cmd': {'type': 'ACTION', 'shape': [6]}}}))
    local = hosts.LocalHost()
    datasets = agent.datasets(str(ws))
    assert list(datasets) == ['cups']
    config = training.new_config(str(ws), 'cups_dp', 'diffusion', 'cups', datasets['cups'])
    assert config['policy']['input_features'].keys() == {'cam', 'joints'} and config['policy']['output_features'].keys() == {'cmd'}
    config_path = f'{training.configs_dir(str(ws))}/cups_dp.json'

    async def configs() -> list:
        assert await training.list_configs(local, str(ws)) == [], 'training_configs/ is created empty'
        await training.write_config(local, config_path, config)
        assert await training.read_config(local, config_path) == config
        return await training.list_configs(local, str(ws))
    assert asyncio.run(configs()) == ['cups_dp']

    # the process side, with a stand-in for `incar train` that logs like incar and honours Ctrl+C
    fake = ws / 'fake_incar'
    fake.write_text(f'''#!{sys.executable}
import signal, sys, time
def log(m): print(time.strftime('INFO %Y-%m-%d %H:%M:%S') + ' ' + '_train.py:361'.rjust(15) + ' ' + m, flush=True)
for i in range(1, 4):  # a tqdm bar: redraws with \\r, newline only at the end
    sys.stdout.write(f'\\rSample dt: {{i * 33}}%|##| {{i}}/3 [00:0{{i}}<00:0{{3 - i}}, 1.00it/s]'); sys.stdout.flush(); time.sleep(0.05)
print(flush=True)
log('Start offline training on a fixed dataset')
try:
    for i in range(1, 1000):
        log(f'step:{{i * 10}} smpl:1 ep:1 epch:0.1 loss:{{1 / i:.3f}} lr:1.0e-04')
        time.sleep(0.1)
except KeyboardInterrupt:
    log('Training interrupted by user')
''')
    fake.chmod(0o755)
    local.incar = str(fake)

    async def process() -> None:
        defaults = {'steps': 100_000, 'log_freq': 200, 'validate_freq': 0}
        cfg_a = {'steps': 10_000, 'log_freq': 10, 'model_save_name': 'a', 'preprocessed_dataset_name': 'cups'}
        cfg_b = {**cfg_a, 'model_save_name': 'b', 'preprocessed_dataset_name': 'cups_b'}
        run = await training.start(local, str(ws), config_path, cfg_a, defaults)
        assert 'already training' in training.conflict(local, cfg_a, str(ws)), 'same policy folder twice'
        assert 'rebuilds' in training.conflict(local, {**cfg_a, 'model_save_name': 'c'}, str(ws)), 'same processed data'
        assert training.conflict(local, cfg_b, str(ws)) is None
        other = await training.start(local, str(ws), config_path, cfg_b, defaults)  # parallel training
        try:
            for _ in range(50):
                await asyncio.sleep(0.1)
                await training.poll()  # all running trainings at once
                if run.progress.step >= 30 and other.progress.step >= 30:
                    break
            assert run.running and run.progress.phase == 'Training' and run.progress.step >= 30, run.progress
            assert other.running and other.progress.step >= 30, other.progress
            bars = [text for _, text, kind in run.lines if kind == 'task']
            assert bars == ['Sample dt: 99%|##| 3/3 [00:03<00:00, 1.00it/s]'], 'only the final bar state is logged'
            training.runs.clear()  # a restarted GUI finds the runs through training_runs/*.json
            found = await training.reattach(local, str(ws))
            assert sorted(r.pid for r in found) == sorted([run.pid, other.pid]) and training.runs == found
            assert training.conflict(local, cfg_a, str(ws)), 'claims come back with the run files'
            assert await training.reattach(local, str(ws)) == [], 'no duplicates'
            for r in found:
                await training.stop(r)
            for _ in range(50):
                await asyncio.sleep(0.1)
                await training.poll()
                if not any(r.running for r in found):
                    break
            for r in found:  # exit code recorded by agent.spawn's wrapper shell
                assert not r.running and r.status() == 'Stopped' and r.returncode == 0, (r.returncode, r.status())
            # past trainings: an ended run rebuilt from its files shows what the live view showed
            info = next(i for i in agent.runs(str(ws)) if i['pid'] == found[0].pid)
            past = await training.load(local, info)
            assert not info['alive'] and past not in training.runs and past.returncode == 0
            assert [t for _, t, _ in past.lines] == [t for _, t, _ in found[0].lines]
            assert past.progress.step == found[0].progress.step and past.progress.loss == found[0].progress.loss
            assert past.ended == info['modified'] and past.ended >= past.started
        finally:
            agent.kill(run.pid, 9)
            agent.kill(other.pid, 9)
    asyncio.run(process())

# remote command lines: quoted for the remote shell, except a leading ~/ which that shell must expand
assert hosts.SshHost.shell(['~/incar/bin/python', '-c', "print('a b')", '{"x": 1}']) == \
    "~/incar/bin/python -c 'print('\"'\"'a b'\"'\"')' '{\"x\": 1}'"

# headset QR must be a standard QR (21+ modules plus 4-module border), not Micro QR which scanners can't read


# headset pairing, as the INCAR GUI does it (headset.py)
def unpack_qr(text: str) -> dict:
    assert text.startswith('xra1:'), text
    body = text[5:]
    return json.loads(__import__('gzip').decompress(base64.urlsafe_b64decode(body + '=' * (-len(body) % 4))))


# the INCAR GUI's own QR payload (from its screenshot) has the shape we produce
incar_qr = unpack_qr('xra1:H4sIAAAAAAAAA02Ky0rDUBQA_-Wsa5Kb-0pml8e5FIoQRMRtW5BWFIMmWCj991537oaZucpekNOyzJSlqQofC1uY2tLE4MvL98PPevg8L7KRL-Eq5znf_7bss3HBRrORNbfX7rRm-Ztxmnbb5_3cL8eP8Wk66OPL0el7jm85pkRyWIc29EocCYpXrBJqzEDtqJV-YIyoJTRUStswBFqP89iWaP4e7fCGmGhHXIVPdJXcbndv0jw02gAAAA')
assert set(incar_qr) == {'a', 'o'} and set(incar_qr['o']) == {'ip', 'p', 'u', 'w', 'f'}


async def headset_pairing() -> None:
    headset.BRIDGE_PORT = hosts.free_port()  # a running GUI may hold the real one
    from aiortc import RTCPeerConnection
    pc = RTCPeerConnection()
    pc.createDataChannel('teleop_command')
    await pc.setLocalDescription(await pc.createOffer())
    offer = pc.localDescription.sdp
    await pc.close()
    packed = unpack_qr(headset.compact_offer(offer, 'http://10.0.0.5:8765/xr-submit'))
    assert set(packed) == set(incar_qr) and set(packed['o']) == set(incar_qr['o']), packed
    assert f'a=ice-ufrag:{packed["o"]["u"]}' in offer and f'a=fingerprint:sha-256 {packed["o"]["f"]}' in offer
    assert any(headset.lan_candidate(line) == (packed['o']['ip'], packed['o']['p']) for line in offer.splitlines())
    assert headset.lan_candidate('a=candidate:1 1 udp 1 10.0.0.5 5000 typ srflx raddr 0.0.0.0 rport 0') is None
    assert "viewBox='0 0 29 29'" not in urllib.parse.unquote(app.headset_qr(headset.compact_offer(offer, 'x'))), 'Micro QR'

    answer = {'type': 'answer', 'sdp': 'v=0\r\n'}
    assert json.loads(headset.answer_line(json.dumps(answer))) == answer
    assert json.loads(headset.answer_line(base64.urlsafe_b64encode(json.dumps(answer).encode()).decode().rstrip('='))) == answer
    try:
        headset.answer_line(json.dumps({'type': 'offer', 'sdp': ''}))
        raise AssertionError('an offer was taken for an answer')
    except ValueError:
        pass

    # the core's side: TcpSocketSignaling server, offer line out, answer line in, socket must stay open
    got, still_open = asyncio.get_running_loop().create_future(), asyncio.get_running_loop().create_future()

    async def core(reader, writer):
        writer.write(json.dumps({'sdp': offer, 'type': 'offer'}).encode() + b'\n')
        await writer.drain()
        got.set_result(json.loads(await reader.readline()))
        try:
            still_open.set_result(await asyncio.wait_for(reader.read(1), 0.5) != b'')
        except asyncio.TimeoutError:
            still_open.set_result(True)
    server = await asyncio.start_server(core, '127.0.0.1', 0)
    bridge = headset.Bridge()
    try:
        qr = unpack_qr(await bridge.pair('127.0.0.1', server.sockets[0].getsockname()[1]))
        assert qr['a'].endswith(f':{headset.BRIDGE_PORT}/xr-submit'), qr
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{headset.BRIDGE_PORT}') as http:
            page = await http.get('/xr-submit')
            assert page.status_code == 200 and headset.COMMANDER_ORIGIN in page.text
            assert (await http.post('/api/manual-answer', content='not an answer')).status_code == 400
            r = await http.post('/api/manual-answer', content=json.dumps(answer))
            assert r.status_code == 200 and r.json() == {'ok': True}, r.text
            assert await asyncio.wait_for(got, 2) == answer
            assert await still_open, 'the bridge closed the signaling socket: the core would drop the headset'
            assert (await http.post('/api/manual-answer', content=json.dumps(answer))).status_code == 400, 'answered twice'
    finally:
        await bridge.stop()
        server.close()


asyncio.run(headset_pairing())

# teleop mapping: keyboard (Unity bindings) and a browser "standard" gamepad
cmd = app.teleop_command({'KeyW', 'KeyA', 'KeyQ', 'Space'}, None, 1.0, 'right')
v = cmd.rightVelocity
assert (v.linear.x, v.linear.y, v.linear.z) == (1, 1, -1) and cmd.rightButtons.gripValue == 1.0
assert cmd.rightButtons.triggerValue == 1.0 and not cmd.HasField('leftButtons'), 'other hand and base must stay unset'
assert app.teleop_command({'KeyW'}, None, 0.0, 'left').leftButtons.gripValue == 0.0, 'no motion without engage'
assert app.teleop_command({'KeyW', 'ShiftLeft'}, None, 0.0, 'left').leftButtons.gripValue == 0.0, 'Shift no longer engages'
assert app.teleop_command({'KeyW'}, None, 0.0, 'left', latched=True).leftButtons.gripValue == 1.0, 'latch engages'
buttons = [0.0] * 17
buttons[0], buttons[7], buttons[4] = 1.0, 0.5, 1.0  # A (engage), RT half, LB
pad = {'axes': [-1.0, -1.0, 0.1, 1.0], 'buttons': buttons}  # stick left+up, right stick x in deadzone, down
v = app.teleop_command(set(), pad, 0.0, 'left').leftVelocity
assert (v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z) == (1, 1, 0.5, 0, 1, -1), v
assert app.teleop_command({'KeyW'}, pad, 0.0, 'left').leftVelocity.linear.x == 1.0, 'keyboard + pad is clipped'


async def teleop_loopback() -> None:
    """Fake core (offers, like incar.application) <- GUI link (answers) carrying one TeleopCommand."""
    from incar_networking.messages.teleop_command_pb2 import TeleopCommand
    from incar_networking.webrtc.webrtc_connection import WebRTCConnection
    from nicegui import core as nicegui_core
    nicegui_core.loop = asyncio.get_running_loop()  # set by ui.run() in the real app; background_tasks needs it
    received = asyncio.get_running_loop().create_future()
    core = WebRTCConnection().add_channel(app.TELEOP_CHANNEL, lambda m: received.done() or received.set_result(m))
    core_task = asyncio.create_task(core.start_connection('127.0.0.1', 19998, send_offer=True))
    assert await app.open_teleop_link('127.0.0.1', 19998)  # starts before the core listens: exercises the retry
    app.teleop.keys = {'Space', 'KeyE'}
    for _ in range(100):
        app.send_teleop()
        if received.done():
            break
        await asyncio.sleep(0.1)
    msg = TeleopCommand.FromString(await asyncio.wait_for(received, 1))
    assert msg.rightVelocity.linear.z == 1.0 and msg.rightButtons.gripValue == 1.0, msg
    await app.disconnect_teleop()
    await core.disconnect()
    core_task.cancel()


asyncio.run(teleop_loopback())

# trainings started outside the GUI: known from their policy folder only
with tempfile.TemporaryDirectory() as outside_ws:
    policy_dir = Path(outside_ws, 'policies', 'cups_act', 'checkpoints')
    for step in ('002500', '005000'):
        Path(policy_dir, step, 'model').mkdir(parents=True)
    Path(policy_dir, 'best', 'training_state').mkdir(parents=True)
    Path(policy_dir, '005000', 'model', 'train_config.json').write_text(json.dumps(
        {'steps': 5000, 'policy': {'type': 'act', 'preprocessing': {'raw_dataset_name': 'cups'}}}))
    Path(policy_dir, 'best', 'training_state', 'training_step.json').write_text('{"step": 2500}')
    Path(policy_dir.parent, 'losses.svg').write_text('<svg/>')
    Path(outside_ws, 'policies', 'no_checkpoints').mkdir()
    # a run file from an older GUI version names only its config; the policy comes from there
    Path(outside_ws, 'training_configs').mkdir()
    Path(outside_ws, 'training_configs', 'gui.json').write_text('{"model_save_name": "gui_policy"}')
    Path(outside_ws, agent.RUNS_DIR).mkdir()
    Path(outside_ws, agent.RUNS_DIR, 'gui_1.json').write_text(json.dumps(
        {'name': 'gui', 'config_path': f'{outside_ws}/training_configs/gui.json', 'log_path': '-', 'pid': 0,
         'started': 0, 'total': 1, 'log_freq': 1, 'validate_freq': 0}))
    assert [r['policy_name'] for r in agent.runs(outside_ws)] == ['gui_policy']
    [info] = agent.trained_policies(outside_ws)
    assert info['name'] == 'cups_act' and [s for s, _ in info['checkpoints']] == [2500, 5000] and info['best'] == 2500
    past = asyncio.run(training.load_policy(hosts.LocalHost(), info))
    assert past.outside and not past.running and past.status() == 'Finished' and past.plot == '<svg/>'
    assert past.progress.step == past.progress.total == 5000 and past.progress.checkpoints == [2500, 5000]
    assert any('act policy on dataset cups' in t for _, t, _ in past.lines), list(past.lines)

# discarding an automatically stopped recording deletes only that saved demo
with tempfile.TemporaryDirectory() as dataset_dir:
    for name in ('demo_3', 'demo_4'):
        Path(dataset_dir, name, 'left.arm').mkdir(parents=True)
    Path(dataset_dir, 'demo_3', 'meta.json').write_text('{}')
    for bad in ('..', 'demo_3/..', 'dataset_config.json', 'demo_4'):  # demo_4: still being saved (no meta.json)
        try:
            agent.remove_demo(dataset_dir, bad)
            raise AssertionError(f'removed {bad}')
        except (ValueError, FileNotFoundError):
            pass
    agent.remove_demo(dataset_dir, 'demo_3')
    assert sorted(p.name for p in Path(dataset_dir).iterdir()) == ['demo_4']

# editing a text feature (the prompt): saved like incar writes it, original kept, wrong input refused
with tempfile.TemporaryDirectory() as feature_dir:
    with h5py.File(Path(feature_dir, 'data.h5'), 'w') as f:  # as incar's _save_feature writes text
        f.create_dataset('timestamps', data=[0, 100_000, 200_000, 300_000])
        f['data'] = ['pick', 'pick', 'place', 'place']
    agent.write_text_series(feature_dir, ['pick', 'lift', 'lift', 'place'])
    assert agent.load_series(feature_dir)['text'] == ['pick', 'lift', 'lift', 'place']
    assert agent.load_series(str(Path(feature_dir)))['t'] == [0.0, 0.1, 0.2, 0.3]
    with h5py.File(Path(feature_dir, 'data.h5.orig'), 'r') as f:
        assert [v.decode() for v in f['data'][:]] == ['pick', 'pick', 'place', 'place'], 'original kept'
    agent.write_text_series(feature_dir, ['a', 'b', 'c', 'd'])
    with h5py.File(Path(feature_dir, 'data.h5.orig'), 'r') as f:
        assert f['data'][0].decode() == 'pick', 'the original is kept once, not overwritten by later saves'
    for bad in (['too', 'few'], ['a', 'b', 'c', 4]):
        try:
            agent.write_text_series(feature_dir, bad)
            raise AssertionError(f'accepted {bad}')
        except ValueError:
            pass
    assert sorted(p.name for p in Path(feature_dir).iterdir()) == ['data.h5', 'data.h5.orig'], 'no temp file left'
with tempfile.TemporaryDirectory() as feature_dir:
    with h5py.File(Path(feature_dir, 'data.h5'), 'w') as f:
        f.create_dataset('timestamps', data=[0, 1])
        f.create_dataset('data', data=np.zeros((2, 3)))
    try:
        agent.write_text_series(feature_dir, ['a', 'b'])
        raise AssertionError('wrote text into a numeric feature')
    except ValueError:
        pass
assert app.text_segments(['a', 'a', 'b', 'a']) == [(0, 1, 'a'), (2, 2, 'b'), (3, 3, 'a')]
assert len(app.text_timeline([0.0], ['a'], (0, 0))['series'][0]['markArea']['data']) == 2  # one segment + selection

# camera manager: incar checks the edited workspace config before the GUI writes it
workspace_config = {'version': '1.0.0', 'cameras': {'wrist': {'type': 'opencv', 'port': 2, 'width': 320}}}
assert asyncio.run(training.check(hosts.LocalHost(), workspace_config, 'workspace')) is None
workspace_config['cameras']['wrist']['type'] = 'no_such_camera'
assert asyncio.run(training.check(hosts.LocalHost(), workspace_config, 'workspace')), 'unknown camera type accepted'
# policy settings editor: incar checks edited policy_config.json contents
assert asyncio.run(training.check(hosts.LocalHost(), {'type': 'act', 'n_action_steps': 10}, 'policy')) is None
assert 'n_action_steps' in asyncio.run(training.check(hosts.LocalHost(), {'type': 'act', 'n_action_steps': 'many'}, 'policy'))
assert asyncio.run(training.check(hosts.LocalHost(), {'type': 'no_such_policy'}, 'policy')), 'unknown policy type accepted'
print('ok')
