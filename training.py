"""Policy training: config files, the schema/check helper, the `incar train` process and its log-based progress.
Everything on the incar PC goes through a host (hosts.py), so it works the same locally and over SSH."""
import asyncio
import base64
import itertools
import json
import os
import re
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agent import RUNS_DIR  # log + run record per training, so a restarted GUI can re-attach
from hosts import marked_json

SCHEMA_SCRIPT = Path(__file__).parent / 'train_schema.py'
SIM_SCRIPT = Path(__file__).parent / 'sim_train.py'
CONFIGS_DIR = 'training_configs'
# preprocessing steps work on the dataset or training samples; steps with only command hooks filter live commands
TRAINING_HOOKS = {'DATASET', 'GET_ITEM', 'GET_ITEM_VALIDATION', 'OBSERVATION', 'POST_INFERENCE'}
NAME = re.compile(r'[A-Za-z0-9][\w.-]*')


# ─── helper process (imports torch + extensions, so never in the GUI process) ─────────────────────────────────

async def run_helper(host, *args: str, stdin: bytes | None = None) -> dict:
    out, err, _ = await host.run_script(SCHEMA_SCRIPT, *args, stdin=stdin)
    return marked_json(out, err)


_schemas: dict[str, dict] = {}  # per incar PC: they can have different extensions
_schema_lock = asyncio.Lock()


async def schema(host) -> dict:
    """Fields, types and defaults of TrainPipelineConfig and every registered policy/optimizer/scheduler/step."""
    async with _schema_lock:
        if host.label not in _schemas:
            _schemas[host.label] = await run_helper(host, 'schema')
    return _schemas[host.label]


async def check(host, config: dict, kind: str = 'train') -> str | None:
    """None if incar accepts the config (kind: 'train', 'dataset' or 'workspace'), else the error."""
    result = await run_helper(host, 'check', '-', kind, stdin=json.dumps(config).encode())
    return None if result['ok'] else result['error']


def training_step_types(schema: dict) -> list[str]:
    return [name for name, fields in schema['registries']['step'].items()
            if set(next((f['default'] for f in fields if f['name'] == 'hooks'), None) or TRAINING_HOOKS) & TRAINING_HOOKS]


# ─── config files and datasets ────────────────────────────────────────────────────────────────────────────────

def configs_dir(workspace: str) -> str:
    return f'{workspace}/{CONFIGS_DIR}'


async def list_configs(host, workspace: str) -> list[str]:
    await host.call('mkdir', configs_dir(workspace))  # the folder is created on first use
    return [name.removesuffix('.json') for name in await host.call('list_names', configs_dir(workspace), '.json')]


async def read_config(host, path: str) -> dict:
    return json.loads(await host.call('read_text', path))


async def write_config(host, path: str, config: dict) -> None:
    await host.call('write_text', path, json.dumps(config, indent=4) + '\n')


def new_config(workspace: str, name: str, policy_type: str, dataset: str, features: dict) -> dict:
    """A trainable starting point: features prefilled from the dataset (state/visual in, actions out)."""
    def pick(types: set) -> dict:
        return {k: {'type': v['type'], 'shape': v['shape']} for k, v in features.items() if v.get('type') in types}

    return {
        'workspace': workspace,
        'model_save_name': name,
        'preprocessed_dataset_name': dataset,
        'policy': {
            'type': policy_type,
            'input_features': pick({'STATE', 'VISUAL'}),
            'output_features': pick({'ACTION'}),
            'preprocessing': {'raw_dataset_name': dataset, 'validation_ratio': 0.15,
                              'steps': [{'type': 'sample_dt', 'dt': 0.1}]},
        },
        'batch_size': 64,
        'steps': 30000,
        'save_freq': 2500,
        'validate_freq': 1000,
    }


# ─── progress from the training log ───────────────────────────────────────────────────────────────────────────

LOG_LINE = re.compile(r'^(INFO|WARNING|ERROR|CRITICAL|DEBUG) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) .{15} (.*)$')
# tqdm bars of the preprocessing steps, e.g. "Sample dt:  71%|███▏ | 25/35 [09:14<04:33, 27.39s/it]"
TQDM = re.compile(r'^(.*?):?\s*(\d+)%\|.*?\|\s*(\d+)/(\d+) \[([\d:]+)<([\d:?]+)')
PHASES = (('Creating datasets', 'Preparing data'), ('Creating policy', 'Building policy'),
          ('Start offline training', 'Training'), ('End of training', 'Finished'),
          ('No improvement for', 'Stopped early'), ('Training interrupted by user', 'Stopped'))
EXCEPTION = re.compile(r'^[\w.]+(Error|Exception|Interrupt)\b')


@dataclass
class Progress:
    """Parsed state of an `incar train` log. Exact steps come from the logging interval: incar prints them rounded."""
    total: int = 0
    log_freq: int = 200
    validate_freq: int = 0
    phase: str = 'Starting'
    step: int = 0
    tracker_lines: int = 0
    metrics: dict = field(default_factory=dict)
    loss: list = field(default_factory=list)  # [step, train loss]
    val: list = field(default_factory=list)  # [step, validation action mse]
    best: tuple | None = None  # (step, validation mse)
    checkpoints: list = field(default_factory=list)
    rate: float | None = None  # steps per second
    _first: tuple | None = None
    error: str | None = None  # last exception line
    task: dict | None = None  # running preprocessing bar: name, done, total, left

    def feed(self, line: str) -> str:
        """Update from one log line; returns its kind for colouring: step, phase, info, warn, error, task or plain."""
        if (bar := TQDM.match(line)) and self.phase != 'Training':
            name, _, done, total, _, left = bar.groups()
            self.task = {'name': name.strip() or 'Preprocessing', 'done': int(done), 'total': int(total), 'left': left}
            return 'task'
        m = LOG_LINE.match(line)
        if not m:
            if EXCEPTION.match(line) or line.startswith('Traceback'):
                if EXCEPTION.match(line):
                    self.error = line
                return 'error'
            return 'plain'
        level, stamp, msg = m.groups()
        if msg.startswith('step:') and ' loss:' in msg:
            self.tracker_lines += 1
            self.step = max(self.step, self.tracker_lines * self.log_freq)
            self.metrics = dict(re.findall(r'(\w+):(\S+)', msg))
            try:
                self.loss.append([self.step, float(self.metrics['loss'])])
            except (KeyError, ValueError):
                pass
            when = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S').timestamp()
            if self._first is None:
                self._first = (when, self.step)
            elif when > self._first[0]:
                self.rate = (self.step - self._first[1]) / (when - self._first[0])
            return 'step'
        if v := re.search(r'action mse of (\S+)', msg):
            # validation runs before that step's progress line, so it belongs to the next validation step
            step = (self.step // self.validate_freq + 1) * self.validate_freq if self.validate_freq else self.step
            try:
                self.val.append([step, float(v[1])])
            except ValueError:
                pass
            return 'info'
        if a := re.search(r'(?:after|at) step (\d+)', msg):
            step = int(a[1])
            self.step = max(self.step, step)
            if msg.startswith('Saving new best model') and self.val:
                self.best = (step, self.val[-1][1])
            elif msg.startswith('Checkpoint policy'):
                self.checkpoints.append(step)
        for prefix, phase in PHASES:
            if msg.startswith(prefix):
                self.phase = phase
                if phase not in ('Preparing data', 'Building policy'):
                    self.task = None
                return 'phase'
        return {'WARNING': 'warn', 'ERROR': 'error', 'CRITICAL': 'error'}.get(level, 'info')

    def eta(self) -> float | None:
        return (self.total - self.step) / self.rate if self.rate and self.total > self.step else None


# ─── the training process ─────────────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingRun:
    host: object  # hosts.LocalHost or hosts.SshHost: where the training runs
    name: str
    log_path: str
    pid: int
    started: float
    progress: Progress
    policy: str = ''  # host:folder the checkpoints go to
    processed: str = ''  # host:folder of the preprocessed dataset it reads
    rebuilds: bool = False  # it deletes and rebuilds `processed` before training
    outside: bool = False  # started outside the GUI: known only from its policy folder (see load_policy)
    plot: str = ''  # incar's losses.svg, shown when there is no log to chart
    returncode: int | None = None  # set once the process ended (-1: exit code unknown)
    ended: float | None = None
    stop_requested: bool = False
    offset: int = 0
    partial: str = ''
    polling: bool = False
    lines: deque = field(default_factory=lambda: deque(maxlen=5000))  # (seq, text, kind)
    seq: itertools.count = field(default_factory=lambda: itertools.count(1))

    @property
    def running(self) -> bool:
        return self.returncode is None

    def status(self) -> str:
        p = self.progress
        if self.running:
            return 'Stopping' if self.stop_requested else p.phase
        if p.phase in ('Finished', 'Stopped early', 'Stopped'):
            return p.phase
        if self.stop_requested:
            return 'Stopped'
        return 'Failed' if self.returncode not in (0, None) or p.error else 'Ended'


runs: list[TrainingRun] = []  # of this GUI session, running and ended
LOG_CHUNK = 4_000_000  # bytes of log per read


def claims(host, config: dict, workspace: str) -> dict:
    """What a training writes or reads that a parallel training must not touch."""
    ws = config.get('workspace') or workspace
    return {'policy': f'{host.label}:{ws}/policies/{config.get("model_save_name")}',
            'processed': f'{host.label}:{ws}/datasets_processed/{config.get("preprocessed_dataset_name")}',
            'rebuilds': not config.get('use_preprocessed_dataset')}


def conflict(host, config: dict, workspace: str) -> str | None:
    """Why this config must not start next to the running trainings, if it must not."""
    mine = claims(host, config, workspace)
    for run in runs:
        if not run.running:
            continue
        if run.policy == mine['policy']:
            return f'"{run.name}" is already training into {mine["policy"].split(":", 1)[1]}.'
        if run.processed == mine['processed'] and (run.rebuilds or mine['rebuilds']):
            return (f'"{run.name}" uses the preprocessed dataset {mine["processed"].split(":", 1)[1]}, and preprocessing '
                    'deletes and rebuilds it. Use another preprocessed_dataset_name.')
    return None


async def start(host, workspace: str, config_path: str, config: dict, defaults: dict,
                simulate: bool = False) -> TrainingRun:
    """Launch `incar train` on the host, detached and logging to a file, so it outlives the GUI.
    simulate runs sim_train.py instead, which only prints a realistic log."""
    base = f'{workspace}/{RUNS_DIR}/{Path(config_path).stem}_{datetime.now():%Y%m%d-%H%M%S-%f}'  # unique per start
    argv = ([host.python, '-P', '-c', SIM_SCRIPT.read_text()] if simulate else [host.incar]) + \
        ['train', '--config_path', config_path]
    env = {'PYTHONUNBUFFERED': '1'}
    if simulate and 'SIM_SECONDS' in os.environ:  # environment variables do not travel over SSH by themselves
        env['SIM_SECONDS'] = os.environ['SIM_SECONDS']
    pid = await host.call('spawn', argv, f'{base}.log', workspace, env)
    progress = Progress(total=config.get('steps', defaults['steps']), log_freq=config.get('log_freq', defaults['log_freq']),
                        validate_freq=config.get('validate_freq', defaults['validate_freq']))
    run = TrainingRun(host=host, name=Path(config_path).stem, log_path=f'{base}.log', pid=pid, started=time.time(),
                      progress=progress, **claims(host, config, workspace))
    runs.append(run)
    await host.call('write_text', f'{base}.json', json.dumps({
        'name': run.name, 'config_path': config_path, 'log_path': run.log_path, 'pid': pid, 'started': run.started,
        'total': progress.total, 'log_freq': progress.log_freq, 'validate_freq': progress.validate_freq,
        **claims(host, config, workspace)}, indent=4))
    return run


def run_from(host, info: dict) -> TrainingRun:
    """A TrainingRun from its run file (see start), its log not read yet."""
    progress = Progress(total=info['total'], log_freq=info['log_freq'], validate_freq=info['validate_freq'])
    return TrainingRun(host=host, name=info['name'], log_path=info['log_path'], pid=info['pid'],
                       started=info['started'], progress=progress, policy=info.get('policy', ''),
                       processed=info.get('processed', ''), rebuilds=info.get('rebuilds', False))


async def reattach(host, workspace: str) -> list[TrainingRun]:
    """Pick up the trainings still running from an earlier GUI session."""
    known = {(r.host.label, r.pid) for r in runs}
    found = [run_from(host, info) for info in await host.call('runs', workspace)
             if info['alive'] and (host.label, info['pid']) not in known]
    runs.extend(found)
    await poll()
    return found


async def load(host, info: dict) -> TrainingRun:
    """An ended training (an entry of the agent's `runs`) with its whole log parsed, for inspection.
    Not added to `runs`: it does not claim anything and never needs polling."""
    run = run_from(host, info)
    while run.running:  # errors reach the caller: a half-read log must not look complete
        state = await host.call('tail', run.log_path, run.offset, LOG_CHUNK, run.pid)
        read(run, {**state, 'alive': False})  # it ended; a live pid now belongs to another process
    run.ended = info.get('modified') or run.ended
    return run


async def load_policy(host, info: dict) -> TrainingRun:
    """A training started outside the GUI (an entry of the agent's `trained_policies`): progress from its checkpoints,
    incar's loss plot, and a summary of the policy folder in place of the log."""
    config, checkpoints = info['config'], info['checkpoints']
    last = checkpoints[-1][0]
    total = config.get('steps') or last
    progress = Progress(total=total, step=last, checkpoints=[s for s, _ in checkpoints],
                        phase='Finished' if last >= total else 'Ended')
    run = TrainingRun(host=host, name=info['name'], log_path=info['folder'], pid=0, started=checkpoints[0][1],
                      progress=progress, outside=True, returncode=0, ended=checkpoints[-1][1])
    if info['plot']:
        run.plot = await host.call('read_text', info['plot'])
    preprocessing = (config.get('policy') or {}).get('preprocessing') or {}
    dataset = preprocessing.get('raw_dataset_name') or config.get('preprocessed_dataset_name') or '?'
    summary = [('This training was started outside the GUI, so there is no log. From its policy folder:', 'phase'),
               (f'{(config.get("policy") or {}).get("type", "?")} policy on dataset {dataset}, {total:,} steps '
                f'planned', 'info'),
               *((f'Checkpoint at step {s:,} · {time.strftime("%d %b %Y, %H:%M", time.localtime(t))}', 'info')
                 for s, t in checkpoints),
               (f'Best checkpoint (lowest validation mse): step {info["best"]:,}' if info['best'] is not None
                else 'No best checkpoint (validation off?)', 'info'),
               ('Times are when checkpoints were written; the training started before the first one.', 'plain')]
    run.lines.extend((next(run.seq), text, kind) for text, kind in summary)
    return run


async def poll() -> None:
    """Read new log output of every running training; called about once a second."""
    await asyncio.gather(*(poll_run(run) for run in runs if run.running))


async def poll_run(run: TrainingRun) -> None:
    """Read new log output and notice when the process ended."""
    if run.polling:
        return
    run.polling = True  # over SSH a call can take longer than the poll interval
    try:
        state = await run.host.call('tail', run.log_path, run.offset, LOG_CHUNK, run.pid)
    except (RuntimeError, OSError):
        return  # e.g. the connection dropped: try again next tick
    finally:
        run.polling = False
    read(run, state)


def read(run: TrainingRun, state: dict) -> None:
    """Feed a `tail` result into the run: log lines, progress, and the exit once the process ended."""
    chunk = base64.b64decode(state['data'])
    run.offset += len(chunk)
    if chunk:
        # progress bars redraw with \r and end with \n: redraws only update progress, the log keeps the final line
        *complete, rest = (run.partial + chunk.decode(errors='replace')).split('\n')
        for line in complete:
            *redraws, final = line.rstrip('\r').split('\r')
            for redraw in redraws:
                run.progress.feed(redraw)
            if final.strip():
                run.lines.append((next(run.seq), final, run.progress.feed(final)))
        *redraws, run.partial = rest.split('\r')
        for redraw in redraws:
            run.progress.feed(redraw)
    if not state['alive']:
        if len(chunk) == LOG_CHUNK:
            return  # more output was written before the end: read it first
        if run.partial.strip():  # a last line without newline
            run.lines.append((next(run.seq), run.partial, run.progress.feed(run.partial)))
        run.partial = ''
        run.returncode = state['exit'] if state['exit'] is not None else -1
        run.ended = time.time()


async def stop(run: TrainingRun, force: bool = False) -> None:
    """Ctrl+C the training (it saves its loss plot and exits); force kills the whole process group."""
    if not run.running:
        return
    run.stop_requested = True
    await run.host.call('kill', run.pid, int(signal.SIGKILL if force else signal.SIGINT))


# categorical colours for several trainings in one chart (dataviz reference palette, dark steps, validated)
SERIES_COLORS = ('#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#0ca30c', '#9085e9', '#e66767')


def combined_chart(runs: list, colors: dict) -> dict:
    """ECharts options: training loss of several trainings over steps, validation dashed; colors: run -> colour."""
    options = loss_chart(Progress(total=max((r.progress.total for r in runs), default=0)))
    values = [v for r in runs for _, v in r.progress.loss + r.progress.val]
    options['yAxis']['type'] = 'log' if values and min(values) > 0 else 'value'
    options['legend']['type'] = 'scroll'
    options['series'] = [series for r in runs for series in (
        {'name': r.name, 'type': 'line', 'data': r.progress.loss, 'showSymbol': False, 'sampling': 'lttb',
         'lineStyle': {'width': 2}, 'itemStyle': {'color': colors[id(r)]}},
        {'name': f'{r.name} validation', 'type': 'line', 'data': r.progress.val, 'symbolSize': 8,
         'lineStyle': {'width': 2, 'type': 'dashed'}, 'itemStyle': {'color': colors[id(r)]}})]
    return options


def loss_chart(p: Progress) -> dict:
    """ECharts options: training loss and validation mse over steps, best validation marked."""
    muted = {'color': '#94a3b8'}
    values = [v for _, v in p.loss + p.val]
    val = {'name': 'validation mse', 'type': 'line', 'data': p.val, 'symbolSize': 7, 'lineStyle': {'width': 2},
           'itemStyle': {'color': '#a78bfa'}}
    if p.best:
        val['markPoint'] = {'data': [{'coord': list(p.best), 'name': 'best', 'value': f'best {p.best[1]:.4g}'}],
                            'symbol': 'circle', 'symbolSize': 12, 'itemStyle': {'color': '#22c55e'},
                            'label': {'position': 'top', 'color': '#86efac', 'fontSize': 11}}
    return {
        'backgroundColor': 'transparent',
        'animation': False,
        'grid': {'left': 60, 'right': 20, 'top': 36, 'bottom': 40},
        'tooltip': {'trigger': 'axis'},
        'legend': {'top': 0, 'textStyle': muted},
        'xAxis': {'type': 'value', 'name': 'step', 'min': 0, 'max': p.total or None, 'axisLabel': muted,
                  'splitLine': {'show': False}},
        # losses span orders of magnitude; a log axis keeps the late, small improvements visible
        'yAxis': {'type': 'log' if values and min(values) > 0 else 'value', 'scale': True, 'axisLabel': muted,
                  'splitLine': {'lineStyle': {'color': 'rgba(148,163,184,0.12)'}}},
        'series': [{'name': 'training loss', 'type': 'line', 'data': p.loss, 'showSymbol': False,
                    'sampling': 'lttb', 'itemStyle': {'color': '#38bdf8'}}, val],
    }
