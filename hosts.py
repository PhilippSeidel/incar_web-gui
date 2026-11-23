"""The PC that runs incar: this one, or a remote PC reached over SSH. Both run the operations in agent.py."""
import asyncio
import json
import os
import shlex
import socket
import time
from pathlib import Path

import agent
import incar_networking

AGENT_SOURCE = Path(agent.__file__).read_text()
LOCAL_VENV = Path(incar_networking.__file__).resolve().parents[4]  # <venv>/lib/python3.x/site-packages/incar_networking


def marked_json(out: bytes, err: bytes) -> dict:
    """The JSON after `@@JSON@@` in a helper's output (incar extensions may print on import)."""
    text = out.decode(errors='replace')
    if '@@JSON@@' not in text:
        raise RuntimeError(err.decode(errors='replace').strip()[-2000:] or 'no answer from the incar PC')
    return json.loads(text.rpartition('@@JSON@@')[2])


def parse_result(out: bytes, err: bytes):
    result = marked_json(out, err)
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result['ok']


# Our code always runs with `python -P`: otherwise the current directory (over SSH: home) comes first on the import
# path, and a project folder named like an installed extension (e.g. ~/image_eq_extension) shadows it.


class LocalHost:
    remote = False
    label = 'this PC'

    def __init__(self):
        self.python, self.incar = str(LOCAL_VENV / 'bin' / 'python'), str(LOCAL_VENV / 'bin' / 'incar')

    async def call(self, op: str, *args):
        return await asyncio.to_thread(agent.OPS[op], *args)

    async def run_script(self, script: Path, *args: str, stdin: bytes | None = None) -> tuple[bytes, bytes, int]:
        """Run one of our scripts with incar's Python (it needs incar's packages)."""
        proc = await asyncio.create_subprocess_exec(self.python, '-P', str(script), *args, stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate(stdin)
        return out, err, proc.returncode


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class SshHost:
    """incar on another PC: one SSH master connection forwards the core's API port and carries every command.
    System ssh client, so ~/.ssh/config, keys and agent apply; no password prompts (BatchMode)."""
    remote = True

    def __init__(self, target: str, venv: str = '~/incar', core_port: int = 8000):
        # extra ssh options (e.g. "-p 2222 -i key") for setups ~/.ssh/config does not cover
        self.ssh_args = shlex.split(os.getenv('INCAR_GUI_SSH_OPTS', ''))
        self.target, self.venv, self.core_port = target, venv.rstrip('/'), core_port
        self.label = target
        self.python, self.incar = f'{self.venv}/bin/python', f'{self.venv}/bin/incar'
        self.local_port = free_port()
        runtime = os.getenv('XDG_RUNTIME_DIR') or '/tmp'
        self.socket = f'{runtime}/incar-gui-{os.getpid()}-{self.local_port}.sock'  # unix sockets: ~100 chars max
        self.master: asyncio.subprocess.Process | None = None

    @property
    def core_url(self) -> str:
        return f'http://127.0.0.1:{self.local_port}'

    def ssh(self, *args: str) -> list[str]:
        return ['ssh', *self.ssh_args, '-o', 'BatchMode=yes', '-o', f'ControlPath={self.socket}', *args]

    @staticmethod
    def shell(argv: list[str]) -> str:
        """The remote command line: everything quoted, except a leading ~/ so the remote shell expands it."""
        return ' '.join('~/' + shlex.quote(a[2:]) if a.startswith('~/') else shlex.quote(a) for a in argv)

    async def connect(self) -> dict:
        self.master = await asyncio.create_subprocess_exec(
            *self.ssh('-M', '-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15',
                      '-L', f'127.0.0.1:{self.local_port}:127.0.0.1:{self.core_port}', self.target),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        for _ in range(150):  # up to 30 s for the login
            if self.master.returncode is not None:
                error = (await self.master.stderr.read()).decode(errors='replace').strip()
                raise ConnectionError(error or f'ssh exited with {self.master.returncode}')
            check = await asyncio.create_subprocess_exec(*self.ssh('-O', 'check', self.target),
                                                         stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            if await check.wait() == 0:
                break
            await asyncio.sleep(0.2)
        else:
            await self.close()
            raise ConnectionError(f'no SSH connection to {self.target} after 30 s')
        try:
            info = await self.call('ping')
        except RuntimeError as e:
            await self.close()
            raise ConnectionError(f'could not run {self.python} on {self.target}: {e}') from None
        if not info['incar']:
            await self.close()
            raise ConnectionError(f'no incar command next to {info["python"]} on {self.target}')
        return info

    async def close(self) -> None:
        if self.master is not None and self.master.returncode is None:
            self.master.terminate()
            await self.master.wait()

    @property
    def connected(self) -> bool:
        return self.master is not None and self.master.returncode is None

    async def run(self, argv: list[str], stdin: bytes | None = None) -> tuple[bytes, bytes, int]:
        proc = await asyncio.create_subprocess_exec(
            *self.ssh(self.target, self.shell(argv)),
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate(stdin)
        return out, err, proc.returncode

    async def call(self, op: str, *args):
        return parse_result(*(await self.run([self.python, '-P', '-c', AGENT_SOURCE, op, json.dumps(args)]))[:2])

    async def run_script(self, script: Path, *args: str, stdin: bytes | None = None) -> tuple[bytes, bytes, int]:
        return await self.run([self.python, '-P', '-c', script.read_text(), *args], stdin)

    async def stream(self, op: str, *args) -> asyncio.subprocess.Process:
        """A long-running agent op whose stdout the caller reads, e.g. the camera frame pump."""
        return await asyncio.create_subprocess_exec(
            *self.ssh(self.target, self.shell([self.python, '-P', '-c', AGENT_SOURCE, op, json.dumps(args)])),
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)


class RemoteFrames:
    """Camera frames from a remote PC: runs agent.pump there and keeps the latest JPEG per stream.
    A stream is live while its frames keep changing; the pump only sends changed frames."""

    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self.streams: list | None = None
        self.frames: dict[str, dict] = {}  # name -> {'jpeg': bytes, 'changed': monotonic time or None}

    async def follow(self, host: SshHost, streams: list) -> None:
        """(Re)start the pump when the stream list changes or the pump died."""
        if streams == self.streams and self.proc is not None and self.proc.returncode is None:
            return
        await self.stop()
        self.streams = streams
        if streams:
            self.proc = await host.stream('pump', streams)
            asyncio.get_running_loop().create_task(self._read(self.proc))

    async def _read(self, proc: asyncio.subprocess.Process) -> None:
        try:
            while True:
                header = json.loads(await proc.stdout.readline())
                jpeg = await proc.stdout.readexactly(header['size'])
                seen = header['name'] in self.frames  # first sight is no evidence of a live feed
                self.frames[header['name']] = {'jpeg': jpeg, 'changed': time.monotonic() if seen else None}
        except (ValueError, asyncio.IncompleteReadError, ConnectionError):
            pass  # pump ended; follow() restarts it

    async def stop(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
            await self.proc.wait()
        self.proc, self.streams = None, None
        self.frames.clear()
