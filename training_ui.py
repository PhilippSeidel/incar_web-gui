"""Training UI: the schema-driven config editor and the progress dashboard (logic lives in training.py)."""
import base64
import copy
import json
import os
import re
import time
from functools import partial
from typing import Awaitable, Callable

from nicegui import ui

import training
from widgets import heading, icon_button, pill, set_pill, set_props, set_style

# top-level fields grouped into form sections; fields not listed here end up in "Other"
SECTIONS = (
    ('Run', 'ph-play-circle', ('model_save_name', 'workspace', 'preprocessed_dataset_name',
                               'use_preprocessed_dataset', 'resume')),
    ('Training', 'ph-gauge', ('batch_size', 'steps', 'seed', 'num_workers', 'dataloading_from_mp4')),
    ('Checkpoints & validation', 'ph-flag-checkered',
     ('save_checkpoint', 'save_freq', 'log_freq', 'validate_freq', 'early_stopping_patience')),
    ('Policy', 'ph-cpu', ('policy',)),
    ('Optimizer & scheduler', 'ph-sliders-horizontal', ('optimizer', 'scheduler')),
    ('Augmentation & sampling', 'ph-magic-wand', ('image_augmentation', 'sparse_sampling_configs')),
)
DATASET_SECTIONS = (
    ('Dataset', 'ph-database', ('video_fps', 'data_timestep', 'routine_array_length', 'version')),
    ('Features', 'ph-list-checks', ('features',)),
    ('Image augmentation', 'ph-magic-wand', ('image_transforms', 'image_noise')),
)
POLICY_SECTIONS = (
    ('Inference', 'ph-lightning', ('device', 'use_amp', 'dt', 'n_obs_steps', 'n_action_steps', 'horizon',
                                   'num_inference_timesteps', 'chunk_size', 'temporal_ensemble_coeff')),
    ('Features', 'ph-list-checks', ('input_features', 'output_features', 'dependent_features', 'normalization_mapping')),
    ('Preprocessing', 'ph-funnel', ('preprocessing',)),
)
HINTS = {
    'video_fps': 'Frame rate of the recorded camera videos',
    'model_save_name': 'Folder under policies/ that receives the checkpoints',
    'workspace': 'Workspace the training reads datasets from and writes the policy to',
    'preprocessed_dataset_name': 'Name of the preprocessed copy in datasets_processed/',
    'use_preprocessed_dataset': 'Reuse an existing preprocessed dataset instead of preprocessing again',
    'resume': 'Continue from the last checkpoint of this model',
    'raw_dataset_name': 'Recorded dataset to train on',
    'validation_ratio': 'Share of demos held out for validation',
    'log_freq': 'Progress is reported every N steps',
    'validate_freq': '0 disables validation',
    'num_workers': 'Data loader processes; 0 loads in the training process',
}
# what each preprocessing step does, from its implementation (incar native steps and the installed extensions)
STEP_HELP = {
    'sample_dt': 'Resamples every demo to a fixed time step: for each multiple of dt it keeps the sample of every '
                 'feature (and the video frame) closest in time, so all features line up. Also sets the video fps to '
                 '1/dt. Most other dataset steps need it to have run first.',
    'downsample_video': 'Resizes the listed camera features to new size [height, width], in the recorded videos and the '
                        'same way on every live camera frame when the policy runs.',
    'filter_by_buttons': 'Keeps only the steps where all listed buttons are pressed above the threshold (joystickValue: '
                         'stick deflection) in any of the listed button features, and cuts the rest from every '
                         'feature. Demos left shorter than the minimum are deleted; takeover demos can be skipped. '
                         'Needs sample_dt before it.',
    'filter_takeover': 'Only for takeover demos (the operator took over while a policy ran): keeps the steps where the '
                       'filter button and the takeover button are both pressed, plus N steps before the first and '
                       'after the last takeover step. Demos without enough steps around the takeover are deleted; '
                       'normal demos are left as they are. Needs sample_dt before it.',
    'remove_takeovers': 'Deletes every takeover demo (the operator took over while a policy ran), keeping only normal '
                        'teleoperation demos.',
    'pad_sampled_signals': 'Widens 0/1 signals of one feature in time: every step above 0.5 also sets the given number '
                           'of steps before and after it to 1, per signal. For fixed-dt data, so after sample_dt.',
    'pad_raw_signals': 'Like pad sampled signals but before sample_dt, on the recording timestamps: values above 0.5 are '
                       'extended by the given seconds before and after. The feature keeps only the signals listed in '
                       'signals to keep and becomes strictly 0/1.',
    'create_validation_set': 'Moves a random share (ratio) of the demos into a separate validation dataset, used for the '
                             'validation loss and the best checkpoint. Usually not needed: a validation ratio in the '
                             'preprocessing settings adds this step automatically.',
    'video_to_h5': 'Decodes every camera video into raw frames stored in an h5 file, so training reads frames without '
                   'decoding: faster loading, much more disk space. The video to h5 option in the preprocessing '
                   'settings adds this step automatically.',
    'image_transform': 'Deprecated and does nothing anymore. Image augmentation (brightness, contrast, saturation, hue, '
                       'noise) now runs on the GPU during training: set it under Augmentation and sampling, image '
                       'augmentation.',
    'histogram_equalize': 'Evens out contrast with CLAHE on the brightness of the listed cameras, baked into the '
                          'recorded videos and applied the same way to every live frame. Clip limit: how strongly '
                          'contrast is boosted (higher also amplifies noise); tile grid: [rows, cols] of regions '
                          'equalized separately.',
    'match_dataset_stats': 'Matches the colour and exposure of live camera frames to the training data. Preprocessing '
                           'only measures the mean and spread of each LAB colour channel of the listed cameras (saved '
                           'to stats path, default ~/.incar/image_eq_dataset_stats.npz) and leaves the dataset as it '
                           'is; when the policy runs, every live frame is shifted onto those values. Strength 0 to 1 '
                           'blends between the original and the full correction.',
    'depth_anything': 'Estimates depth for every frame of the source camera with the Depth Anything 3 model and stores '
                      'it as a new camera feature (target feature name) in grey, scaled to the deepest value in the '
                      'dataset. Also runs on live frames when the policy runs. Needs the S2 extra; slow without a GPU.',
    'grounded_SAM': 'Finds the objects named in prompts (Grounding DINO) and cuts them out (SAM 2), stored as a new '
                    'camera feature (target feature name): the objects painted in the mask colour and everything else '
                    'in the inverse colour (each optional), the mask grown by the dilation pixels. Can also run on '
                    'live frames. Needs the S2 extra; heavy on the GPU.',
    'combine_single_channels': 'Builds a new camera feature (target feature name) from the frames of the first '
                               'feature, with its red channel replaced by the red channel of the second feature and, '
                               'if set, its green channel by that of the third; blue stays from the first. Meant for '
                               'single-channel images such as depth maps or masks. Also applied to live frames.',
    'lerobot_v2_conversion': 'Exports the preprocessed dataset in LeRobot v2.1 format (parquet and mp4) to '
                             'datasets_lerobot in the workspace, for policies trained outside incar such as GR00T; '
                             'task is the instruction stored with every episode. Needs sample_dt before it and the '
                             'gr00t extra.',
}
DATASET_FIELDS = {'raw_dataset_name'}
FEATURE_FIELDS = re.compile(r'(^|_)(feature|feature_name|data_key)$')


def humanize(name: str) -> str:
    return name.replace('_', ' ')


def parse_numbers(text: str, kind: str) -> list:
    parts = [p for p in re.split(r'[,\s]+', text.strip().strip('[]')) if p]
    return [int(p) if kind == 'int' else float(p) for p in parts]


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return '–'
    seconds = int(seconds)
    h, m, s = seconds // 3600, seconds // 60 % 60, seconds % 60
    return f'{h}h {m:02d}m' if h else f'{m}m {s:02d}s' if m else f'{s}s'


async def confirm(text: str, ok: str = 'Continue', cancel: str = 'Cancel') -> bool:
    with ui.dialog() as dialog, ui.card().classes('gap-4 p-6 max-w-md'):
        ui.label(text)
        with ui.row().classes('w-full justify-end'):
            ui.button(cancel, on_click=lambda: dialog.submit(False)).props('flat')
            ui.button(ok, on_click=lambda: dialog.submit(True))
    result = await dialog
    dialog.delete()
    return bool(result)


def note(banner: ui.element, text: str, state: str) -> None:
    with banner, ui.element('div').classes('banner w-full').props(f'data-state={state}'):
        ui.icon('ph-warning' if state != 'ok' else 'ph-check-circle')
        ui.label(text).classes('whitespace-pre-wrap')


async def incar_check(banner: ui.element, host, config: dict, kind: str) -> str | None:
    """incar's verdict on a config (None: accepted), with a spinner in banner meanwhile."""
    banner.clear()
    with banner, ui.row().classes('items-center gap-2') as checking:
        ui.spinner()
        ui.label('Checking the config with incar…').classes('hint')
    try:
        error = await training.check(host, config, kind)
    except Exception as e:
        error = f'The check itself failed: {e}'
    checking.delete()
    return error


class ConfigForm:
    """A training (kind='train'), dataset (kind='dataset') or policy (kind='policy', schema['policy']: the fields of
    its type) config dict as a form (CameraManager uses fields()). Fields show the config value or
    the schema default; only values the user sets end up in the dict (and the file)."""

    def __init__(self, schema: dict, datasets: dict, on_change: Callable[[], None], kind: str = 'train',
                 live_features: dict | None = None):
        self.schema, self.datasets, self.on_change, self.kind = schema, datasets, on_change, kind
        self.live_features = live_features  # feature suggestions for a new dataset: what the core sees right now
        self.data: dict = {}
        self.container = ui.column().classes('w-full gap-4')

    def load(self, data: dict) -> None:
        self.data = data
        self.rebuild()

    def rebuild(self) -> None:
        self.container.clear()
        fields = {f['name']: f for f in self.schema[self.kind]}
        base = {'train': SECTIONS, 'dataset': DATASET_SECTIONS}.get(self.kind, POLICY_SECTIONS)
        listed = {name for *_, names in base for name in names}
        sections = [*base, ('Other', 'ph-dots-three', tuple(n for n in fields if n not in listed))]
        with self.container:
            for title, icon, names in sections:
                if any(n in fields for n in names):
                    with ui.element('div').classes('form-section w-full'):
                        heading(title, icon)
                        self.fields([fields[n] for n in names if n in fields], self.data, lambda: None)

    # ── dataset context for suggestions ──
    def dataset_name(self) -> str:
        preprocessing = (self.data.get('policy') or {}).get('preprocessing') or {}
        return preprocessing.get('raw_dataset_name') or self.data.get('preprocessed_dataset_name') or ''

    def dataset_specs(self) -> dict:
        return self.live_features if self.live_features is not None else self.datasets.get(self.dataset_name(), {})

    def feature_names(self) -> list[str]:
        policy = self.data.get('policy') or {}
        names = set(self.dataset_specs())
        for key in ('input_features', 'output_features', 'dependent_features'):
            names |= set(policy.get(key) or {})
        return sorted(names)

    # ── rendering ──
    @staticmethod
    def simple(f: dict) -> bool:
        return f['kind'] in ('bool', 'int', 'float', 'str', 'enum') or (
            f['kind'] == 'list' and f['item']['kind'] in ('str', 'int', 'float', 'enum'))

    def fields(self, fields: list, data: dict, touch: Callable[[], None]) -> None:
        simple = [f for f in fields if self.simple(f) and f['name'] != 'type']
        if simple:
            with ui.element('div').classes('w-full grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-x-4 gap-y-3'):
                for f in simple:
                    self.field(f, data, touch)
        for f in fields:
            if not self.simple(f) and f['name'] != 'type':
                self.field(f, data, touch)

    def field(self, f: dict, data: dict, touch: Callable[[], None]) -> None:
        name, kind = f['name'], f['kind']
        value = data[name] if name in data else f['default']
        label = humanize(name) + (' *' if f.get('required') else '')

        def put(v, rebuild: bool = False) -> None:
            data[name] = v
            touch()
            self.on_change()
            if rebuild:
                self.rebuild()

        widget = None
        if kind == 'bool':
            widget = ui.switch(label, value=bool(value), on_change=lambda e: put(e.value))
        elif kind in ('int', 'float') and (isinstance(value, bool) or not isinstance(value, (int, float, type(None)))):
            self.json_field(f, data, put)  # e.g. text typed into the JSON tab: show it, incar's check names the error
        elif kind in ('int', 'float'):
            def number(e) -> None:
                if e.value is None:
                    if f.get('nullable'):
                        put(None)
                    return
                put(int(e.value) if kind == 'int' else float(e.value))
            widget = ui.number(label, value=value, precision=0 if kind == 'int' else None, on_change=number)
            if f.get('nullable'):
                widget.props('clearable')
        elif kind == 'str' and name in DATASET_FIELDS:
            # a dataset that is not (or no longer) in the workspace stays visible instead of breaking the form
            widget = ui.select(sorted(set(self.datasets) | ({value} if value else set())), value=value or None, label=label,
                               new_value_mode='add-unique', on_change=lambda e: put(e.value or '', rebuild=True))
        elif kind == 'str' and FEATURE_FIELDS.search(name):
            widget = ui.select(self.feature_names(), value=value or None, label=label, new_value_mode='add-unique',
                               clearable=True, on_change=lambda e: put(e.value or ''))
        elif kind == 'str':
            widget = ui.input(label, value='' if value is None else str(value),
                              on_change=lambda e: put(e.value if e.value or not f.get('nullable') else None))
        elif kind == 'enum':
            widget = ui.select(self.schema['enums'][f['enum']], value=value, label=label,
                               clearable=bool(f.get('nullable')), on_change=lambda e: put(e.value))
        elif kind == 'list' and f['item']['kind'] in ('str', 'enum'):
            options = (self.schema['enums'][f['item']['enum']] if f['item']['kind'] == 'enum'
                       else sorted(set(self.feature_names()) | set(value or [])))
            widget = ui.select(options, value=list(value or []), label=label, multiple=True,
                               new_value_mode='add-unique' if f['item']['kind'] == 'str' else None,
                               on_change=lambda e: put(list(e.value or []))).props('use-chips')
        elif kind == 'list' and f['item']['kind'] in ('int', 'float'):
            text = '' if value is None else ', '.join(str(v) for v in value)

            def numbers(e) -> None:
                try:
                    parsed = parse_numbers(e.value, f['item']['kind'])
                except ValueError:
                    e.sender.props('error error-message="Numbers separated by commas"')
                    return
                if f.get('length') and parsed and len(parsed) != f['length']:
                    e.sender.props(f'error error-message="Exactly {f["length"]} numbers"')
                    return
                e.sender.props(remove='error')
                put(parsed if parsed or not f.get('nullable') else None)
            widget = ui.input(label, value=text, placeholder='e.g. 216, 288', on_change=numbers)
        elif kind == 'list' and f['item']['kind'] == 'choice':
            self.choice_list(f, data, touch)
        elif kind == 'list' and f['item']['kind'] == 'struct':
            self.struct_list(f, data, touch)
        elif kind == 'struct':
            self.struct(f, data, touch)
        elif kind == 'choice':
            self.choice(f, data, touch)
        elif kind == 'dict' and f['value'].get('struct') == 'PolicyFeature':
            self.features(f, data, touch)
        elif kind == 'dict' and f['key']['kind'] == 'enum' and f['value']['kind'] == 'enum':
            self.enum_map(f, data, touch)
        else:
            self.json_field(f, data, put)
        if widget is not None and name in HINTS:
            widget.tooltip(HINTS[name]) if kind == 'bool' else widget.props(f'hint="{HINTS[name]}"')

    def attached(self, data: dict, name: str, empty, touch: Callable[[], None]):
        """The nested value for `name` and a touch() that puts it into `data` once something inside changes."""
        sub = data[name] if isinstance(data.get(name), type(empty)) else empty

        def touch_sub() -> None:
            if data.get(name) is not sub:
                data[name] = sub
                touch()
        return sub, touch_sub

    def mutate(self, change: Callable[[], None], attach: Callable[[], None]) -> None:
        """A structural change (add, remove, reorder): apply, mark changed, redraw."""
        change()
        attach()
        self.on_change()
        self.rebuild()

    def sub(self, title: str):
        return ui.expansion(title, value=True).classes('w-full form-sub').props('dense')

    def struct(self, f: dict, data: dict, touch) -> None:
        sub, touch_sub = self.attached(data, f['name'], {}, touch)
        with self.sub(humanize(f['name'])):
            self.fields(self.schema['structs'][f['struct']], sub, touch_sub)

    def choice(self, f: dict, data: dict, touch) -> None:
        name, registry = f['name'], self.schema['registries'][f['registry']]
        sub = data.get(name) if isinstance(data.get(name), dict) else None
        current = (sub or {}).get('type', '')
        options = ({'': 'Policy default'} if f.get('nullable') and name != 'policy' else {}) | {t: t for t in registry}
        if current and current not in registry:
            options[current] = f'{current} (not installed)'

        def set_type(e) -> None:
            if not e.value:
                data.pop(name, None)
            else:  # keep what both types understand, e.g. features and preprocessing when switching policies
                keep = {k: v for k, v in (sub or {}).items() if k in {x['name'] for x in registry[e.value]}}
                data[name] = {**keep, 'type': e.value}
            touch()
            self.on_change()
            self.rebuild()

        with self.sub(humanize(name)):
            ui.select(options, value=current, label=f'{humanize(name)} type', on_change=set_type).classes('w-72')
            if sub is not None and current in registry:
                self.fields(registry[current], sub, touch)
            elif sub is not None:
                self.json_field({'name': name, 'default': sub}, data, lambda v: (data.__setitem__(name, v), touch(),
                                                                                 self.on_change()))

    def choice_list(self, f: dict, data: dict, touch) -> None:
        name, key = f['name'], f['item']['registry']
        registry = self.schema['registries'][key]
        items, attach = self.attached(data, name, [], touch)
        allowed = training.training_step_types(self.schema) if key == 'step' else list(registry)
        mutate = partial(self.mutate, attach=attach)

        with self.sub(humanize(name)):
            if not items:
                ui.label('No steps yet').classes('hint')
            for i, item in enumerate(items):
                kind = item.get('type', '?')
                with ui.element('div').classes('list-item w-full'):
                    with ui.row().classes('w-full items-center gap-2 no-wrap'):
                        ui.label(str(i + 1)).classes('list-index')
                        kind_label = ui.label(kind).classes('font-medium')
                        if kind in STEP_HELP:
                            kind_label.tooltip(STEP_HELP[kind]).classes('cursor-help')
                        if kind not in registry:
                            ui.label('not installed').classes('pill text-xs').props('data-state=bad')
                        ui.space()
                        if i > 0:
                            icon_button('ph-arrow-up', 'Move up', lambda i=i: mutate(lambda: items.insert(i - 1, items.pop(i))))
                        if i < len(items) - 1:
                            icon_button('ph-arrow-down', 'Move down',
                                        lambda i=i: mutate(lambda: items.insert(i + 1, items.pop(i))))
                        icon_button('ph-trash', 'Remove', lambda i=i: mutate(lambda: items.pop(i)))
                    if kind in registry:
                        self.fields(registry[kind], item, attach)
                    else:
                        ui.label(f'"{kind}" is not provided by any installed extension; '
                                 '`incar train` will reject this config.').classes('hint text-negative')
            with ui.row().classes('items-center gap-2'):
                new_type = ui.select(allowed, label='Step type', value=allowed[0] if allowed else None).classes('w-64')
                if key == 'step':  # what a step does, on hover over its option
                    new_type.add_slot('option', '''
                        <q-item v-bind="props.itemProps">
                          <q-item-section><q-item-label>{{ props.opt.label }}</q-item-label></q-item-section>
                          <q-tooltip anchor="center right" self="center left" max-width="380px" class="text-sm">
                            {{ (''' + json.dumps(STEP_HELP) + ''')[props.opt.label] || 'No description for this step.' }}
                          </q-tooltip>
                        </q-item>''')
                ui.button('Add step', icon='ph-plus-circle',
                          on_click=lambda: new_type.value and mutate(lambda: items.append({'type': new_type.value}))
                          ).props('flat')

    def struct_list(self, f: dict, data: dict, touch) -> None:
        items, attach = self.attached(data, f['name'], [], touch)
        fields = self.schema['structs'][f['item']['struct']]
        mutate = partial(self.mutate, attach=attach)

        with self.sub(humanize(f['name'])):
            for i, item in enumerate(items):
                with ui.element('div').classes('list-item w-full'):
                    with ui.row().classes('w-full items-center no-wrap'):
                        ui.label(str(i + 1)).classes('list-index')
                        ui.space()
                        icon_button('ph-trash', 'Remove', lambda i=i: mutate(lambda: items.pop(i)))
                    self.fields(fields, item, attach)
            ui.button('Add', icon='ph-plus-circle', on_click=lambda: mutate(lambda: items.append({}))).props('flat')

    def features(self, f: dict, data: dict, touch) -> None:
        feats, attach = self.attached(data, f['name'], {}, touch)
        specs = self.dataset_specs()
        types, modes = self.schema['enums']['FeatureType'], self.schema['enums']['NormalizationMode']

        def changed() -> None:
            attach()
            self.on_change()
        mutate = partial(self.mutate, attach=attach)

        with self.sub(humanize(f['name'])):
            for fname, spec in feats.items():
                with ui.row().classes('w-full items-center gap-2 no-wrap'):
                    ui.label(fname).classes('font-mono text-sm grow min-w-0 truncate').tooltip(fname)
                    ui.select(types, value=spec.get('type'), label='type',
                              on_change=lambda e, s=spec: (s.__setitem__('type', e.value), changed())).classes('w-28')
                    shape = ui.input('shape', value=', '.join(str(v) for v in spec.get('shape', []))).classes('w-32')

                    def set_shape(e, s=spec) -> None:
                        try:
                            s['shape'] = parse_numbers(e.value, 'int')
                            e.sender.props(remove='error')
                            changed()
                        except ValueError:
                            e.sender.props('error')
                    shape.on_value_change(set_shape)
                    ui.select(modes, value=spec.get('normalization', 'NULL'), label='normalization',
                              on_change=lambda e, s=spec: (s.__setitem__('normalization', e.value), changed())
                              ).classes('w-36').tooltip('NULL: use the policy\'s normalization mapping')
                    icon_button('ph-trash', 'Remove', lambda n=fname: mutate(lambda: feats.pop(n)))
            if not feats:
                ui.label('No features').classes('hint')
            candidates = [n for n in specs if n not in feats]
            with ui.row().classes('items-center gap-2'):
                pick = ui.select(candidates, label='Add feature', new_value_mode='add-unique',
                                 value=candidates[0] if candidates else None).classes('w-80')

                def add() -> None:
                    if pick.value and pick.value not in feats:
                        spec = specs.get(pick.value, {'type': 'STATE', 'shape': [1]})
                        mutate(lambda: feats.update({pick.value: {'type': spec['type'], 'shape': list(spec['shape'])}}))
                ui.button('Add', icon='ph-plus-circle', on_click=add).props('flat')
            if self.live_features is not None:
                ui.label('Suggestions are the features the core currently receives and sends.').classes('hint')
            elif specs:
                ui.label(f'Suggestions and shapes come from dataset "{self.dataset_name()}". Adjust shapes '
                         'that preprocessing changes, e.g. after downsample_video.').classes('hint')

    def enum_map(self, f: dict, data: dict, touch) -> None:
        mapping, attach = self.attached(data, f['name'], {}, touch)
        default = f['default'] or {}
        values = self.schema['enums'][f['value']['enum']]
        with self.sub(humanize(f['name'])):
            with ui.element('div').classes('w-full grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-3'):
                for key in self.schema['enums'][f['key']['enum']]:
                    def set_value(e, key=key) -> None:
                        if e.value:
                            mapping[key] = e.value
                        else:
                            mapping.pop(key, None)
                        attach()
                        self.on_change()
                    ui.select({'': f'default ({default.get(key, "none")})', **{v: v for v in values}},
                              value=mapping.get(key, ''), label=key, on_change=set_value)

    def json_field(self, f: dict, data: dict, put: Callable) -> None:
        value = data.get(f['name'], f['default'])
        area = ui.textarea(humanize(f['name']) + ' (JSON)', value=json.dumps(value, indent=2)) \
            .classes('w-full font-mono').props('autogrow')

        def parse() -> None:
            try:
                parsed = json.loads(area.value)
            except ValueError:
                area.props('error error-message="Invalid JSON"')
                return
            area.props(remove='error')
            if parsed != value:
                put(parsed)
        area.on('blur', parse)


class FormJson:
    """Form and JSON tabs over one config dict, kept in sync; on_edit runs when the JSON tab changed the dict."""

    def __init__(self, on_edit: Callable[[], None]):
        self.on_edit = on_edit
        self.form: ConfigForm | None = None
        self.silent = self.json_active = False
        with ui.tabs(on_change=self.tab_changed).props('dense no-caps inline-label align=left') as self.tabs:
            ui.tab('form', 'Form', icon='ph-check-square-offset')
            ui.tab('json', 'JSON', icon='ph-brackets-curly')
        with ui.tab_panels(self.tabs, value='form').classes('w-full grow min-h-0'):
            with ui.tab_panel('form').classes('p-0 overflow-y-auto h-full pr-2'):
                self.area = ui.column().classes('w-full')
            with ui.tab_panel('json').classes('p-0 h-full'):
                self.editor = ui.codemirror(language='JSON', theme='githubDark').classes('w-full h-full')

    def show_form(self) -> None:
        self.set_tab('form')
        self.json_active = False

    def set_tab(self, tab: str) -> None:
        self.silent = True  # programmatic switch: no syncing in either direction
        self.tabs.set_value(tab)
        self.silent = False

    def tab_changed(self, e) -> None:
        if self.form is None or self.silent:
            return
        if e.value == 'json':
            self.editor.set_value(json.dumps(self.form.data, indent=4))
            self.json_active = True
        elif self.json_active:
            if self.apply_json():
                self.json_active = False
            else:
                self.set_tab('json')  # stay, keeping the user's (invalid) text

    def sync(self) -> bool:
        """Take pending edits from the JSON tab; False (and a notice) if it does not parse."""
        return self.apply_json() if self.json_active else True

    def apply_json(self) -> bool:
        try:
            parsed = json.loads(self.editor.value)
        except ValueError as e:
            ui.notify(f'The JSON is not valid: {e}', type='negative')
            return False
        if not isinstance(parsed, dict):
            ui.notify('The config must be a JSON object', type='negative')
            return False
        if parsed != self.form.data:
            self.form.load(parsed)
            self.on_edit()
        return True


class Trainer:
    """Dialog: pick, create, edit, validate and save training configs, then start a training."""

    def __init__(self, context: Callable[[], Awaitable[tuple | None]], show_progress: Callable):
        """context() returns (host, workspace) of the connected incar PC, or None if it is unreachable."""
        self.get_context, self.show_progress = context, show_progress
        self.host = None
        self.workspace: str | None = None
        self.schema: dict | None = None
        self.datasets: dict = {}
        self.names: list[str] = []
        self.name: str | None = None
        self.dirty = False
        with ui.dialog().props('maximized') as self.dialog, ui.card().classes('w-full h-full gap-4 no-wrap p-5'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                heading('Train a policy', 'ph-graduation-cap')
                self.location = ui.label().classes('hint')
                ui.space()
                icon_button('ph-x', 'Close', self.close)
            with ui.row().classes('w-full grow no-wrap gap-4 min-h-0'):
                with ui.column().classes('w-72 shrink-0 h-full glass p-3 gap-2 no-wrap'):
                    ui.label('Training configs').classes('section-label')
                    self.config_list = ui.list().classes('w-full grow overflow-y-auto')
                    with ui.row().classes('w-full gap-2'):
                        ui.button('New', icon='ph-plus-circle', on_click=lambda: self.create(copy_current=False)).props('flat')
                        ui.button('Duplicate', icon='ph-copy', on_click=lambda: self.create(copy_current=True)).props('flat')
                with ui.column().classes('grow h-full min-w-0 gap-3 no-wrap'):
                    with ui.row().classes('w-full items-center gap-3'):
                        self.title = ui.label().classes('text-lg font-medium')
                        self.dirty_pill = ui.label('unsaved changes').classes('pill text-xs').props('data-state=warn')
                        ui.space()
                        ui.button('Validate', icon='ph-check-circle', on_click=self.validate).props('flat')
                        ui.button('Save', icon='ph-floppy-disk', on_click=self.save).props('flat')
                        ui.button('Start training', icon='ph-play', on_click=self.start).props('color=positive')
                    self.banner = ui.column().classes('w-full gap-2')
                    self.tabs = FormJson(lambda: self.set_dirty(True))
                    self.form_area = self.tabs.area
        self.dirty_pill.set_visibility(False)

    # ── lifecycle ──
    async def open(self) -> None:
        context = await self.get_context()
        if context is None:
            return
        host, workspace = context
        if (host, workspace) != (self.host, self.workspace):
            self.name = None  # another incar PC or workspace: start from its first config
        self.host, self.workspace = host, workspace
        self.location.set_text(f'{host.label}: {training.configs_dir(workspace)}')
        self.dialog.open()
        self.form_area.clear()
        with self.form_area:
            with ui.row().classes('items-center gap-3 p-4'):
                ui.spinner(size='md')
                ui.label(f'Loading configs and the config schema from incar on {host.label}…')
        try:
            self.schema = await training.schema(host)  # first time per PC: imports torch there, a few seconds
            self.datasets = await host.call('datasets', workspace)
            self.names = await training.list_configs(host, workspace)  # creates training_configs/ if missing
        except Exception as e:  # show why instead of an empty form
            self.form_area.clear()
            with self.form_area:
                ui.label(f'Could not load from {host.label}: {e}').classes('text-negative')
            return
        self.form_area.clear()
        with self.form_area:
            self.empty = ui.label('Create a config with "New" to get started.').classes('hint p-4')
            self.tabs.form = ConfigForm(self.schema, self.datasets, lambda: self.set_dirty(True))
        if self.name not in self.names:
            self.name = self.names[0] if self.names else None
        self.render_list()
        await self.load(self.name) if self.name else self.show_empty()

    async def close(self) -> None:
        if self.dirty and not await confirm('Discard unsaved changes?', 'Discard'):
            return
        self.dirty = False
        self.dialog.close()

    @property
    def form(self) -> ConfigForm | None:
        return self.tabs.form

    def show_empty(self) -> None:
        self.title.set_text('No config yet')
        self.empty.set_visibility(True)
        self.form.container.set_visibility(False)

    def render_list(self) -> None:
        self.config_list.clear()
        with self.config_list:
            for name in self.names:
                item = ui.item(on_click=lambda n=name: self.switch(n)).classes('demo-item')
                with item:
                    with ui.item_section():
                        ui.item_label(name)
                if name == self.name:
                    item.classes('demo-active')

    async def switch(self, name: str) -> None:
        if name == self.name or (self.dirty and not await confirm('Discard unsaved changes?', 'Discard')):
            return
        await self.load(name)

    def path(self, name: str | None = None) -> str:
        return f'{training.configs_dir(self.workspace)}/{name or self.name}.json'

    async def load(self, name: str) -> None:
        try:
            data = await training.read_config(self.host, self.path(name))
        except (RuntimeError, OSError, ValueError) as e:
            ui.notify(f'Cannot read {name}.json: {e}', type='negative')
            return
        self.name = name
        self.title.set_text(f'{name}.json')
        self.banner.clear()
        self.tabs.show_form()
        self.empty.set_visibility(False)
        self.form.container.set_visibility(True)
        self.form.load(data)
        self.set_dirty(False)
        self.render_list()
        self.check_workspace()

    def set_dirty(self, dirty: bool) -> None:
        self.dirty = dirty
        self.dirty_pill.set_visibility(dirty)

    def check_workspace(self) -> None:
        configured = (self.form.data.get('workspace') or '').rstrip('/')
        current = self.workspace.rstrip('/')
        # the config may use ~ for the home folder on the incar PC, which this GUI cannot expand
        if configured and configured != current and not (configured.startswith('~/') and current.endswith(configured[1:])):
            self.note(f'This config trains in {configured}, not in the current workspace {current}. '
                      'incar train also makes that workspace the saved default.', 'warn')

    def note(self, text: str, state: str) -> None:
        note(self.banner, text, state)

    # ── actions ──
    async def create(self, copy_current: bool) -> None:
        if self.form is None or (copy_current and not self.name):
            return
        policies = list(self.schema['registries']['policy'])
        datasets = sorted(self.datasets)
        with ui.dialog() as dialog, ui.card().classes('gap-3 p-6 w-96'):
            heading('Duplicate config' if copy_current else 'New config', 'ph-plus-circle')
            name = ui.input('Name', placeholder='e.g. cups_diffusion').classes('w-full')
            if not copy_current:
                policy = ui.select(policies, value='diffusion' if 'diffusion' in policies else policies[0],
                                   label='Policy type').classes('w-full')
                dataset = ui.select(datasets, value=datasets[0] if datasets else None, label='Dataset').classes('w-full')
                ui.label('Inputs (state, visual) and outputs (actions) are prefilled from the dataset.').classes('hint')
            with ui.row().classes('w-full justify-end'):
                ui.button('Cancel', on_click=lambda: dialog.submit(None)).props('flat')
                ui.button('Create', on_click=lambda: dialog.submit(name.value.strip()))
        new = await dialog
        dialog.delete()
        if not new:
            return
        if not training.NAME.fullmatch(new) or new.endswith('.json'):
            ui.notify('Use letters, digits, ".", "_" or "-" for the name', type='negative')
            return
        if new in self.names:
            ui.notify(f'{new}.json already exists', type='negative')
            return
        if copy_current and not self.tabs.sync():
            return
        data = ({**copy.deepcopy(self.form.data), 'model_save_name': new} if copy_current else training.new_config(
            self.workspace, new, policy.value, dataset.value or '', self.datasets.get(dataset.value, {})))
        await training.write_config(self.host, self.path(new), data)
        self.names = sorted([*self.names, new])
        self.dirty = False
        await self.load(new)

    async def save(self) -> bool:
        if self.form is None or self.name is None or not self.tabs.sync():
            return False
        try:
            await training.write_config(self.host, self.path(), self.form.data)
        except (RuntimeError, OSError) as e:
            ui.notify(f'Could not save: {e}', type='negative')
            return False
        self.set_dirty(False)
        ui.notify(f'Saved {self.name}.json', type='positive')
        return True

    async def run_check(self) -> bool:
        if self.form is None or self.name is None or not self.tabs.sync():
            return False
        error = await incar_check(self.banner, self.host, self.form.data, 'train')  # what is on screen, saved or not
        self.check_workspace()
        if error:
            self.note(error, 'bad')
            return False
        return True

    async def validate(self) -> None:
        if await self.run_check():
            self.note('incar accepts this config.', 'ok')

    async def start(self) -> None:
        if not await self.run_check():
            return
        data = self.form.data
        if reason := training.conflict(self.host, data, self.workspace):
            self.note(f'Not started: {reason}', 'bad')
            return
        model = data.get('model_save_name') or ''
        workspace = data.get('workspace') or self.workspace
        if not data.get('resume') and await self.host.call('exists', f'{workspace}/policies/{model}/checkpoints'):
            if not await confirm(f'policies/{model} already has checkpoints. Training again overwrites them. '
                                 'Set "resume" to continue it instead.', 'Overwrite'):
                return
        if self.dirty and not await self.save():
            return
        defaults = {f['name']: f['default'] for f in self.schema['train']}
        try:
            run = await training.start(self.host, self.workspace, self.path(), data, defaults,
                                       simulate=os.getenv('TRAIN_SIMULATION') == '1')
        except (RuntimeError, OSError) as e:
            ui.notify(f'Could not start incar train: {e}', type='negative')
            return
        self.dialog.close()
        self.show_progress(run)


class DatasetCreator:
    """Dialog: configure a new dataset (form + JSON), check it with incar, create it through the core."""

    def __init__(self, context: Callable[[], Awaitable[tuple | None]], live_features: Callable[[], dict],
                 existing: Callable[[], list], create: Callable[[str, dict], Awaitable[bool]]):
        self.get_context, self.live_features, self.existing, self.create_dataset = context, live_features, existing, create
        self.host = None
        with ui.dialog().props('maximized') as self.dialog, ui.card().classes('w-full h-full gap-4 no-wrap p-5'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                heading('New dataset', 'ph-folder-simple-plus')
                ui.space()
                icon_button('ph-x', 'Close', self.dialog.close)
            with ui.row().classes('w-full items-start gap-3'):
                self.name = ui.input('Dataset name', placeholder='e.g. cups_left_arm').classes('w-80') \
                    .props('hint="Folder under datasets/ in the current workspace"')
                ui.space()
                ui.button('Validate', icon='ph-check-circle', on_click=self.validate).props('flat')
                ui.button('Create dataset', icon='ph-plus-circle', on_click=self.create).props('color=positive')
            self.banner = ui.column().classes('w-full gap-2')
            self.tabs = FormJson(lambda: None)

    async def open(self) -> None:
        context = await self.get_context()
        if context is None:
            return
        self.host = context[0]
        self.name.set_value('')
        self.banner.clear()
        self.dialog.open()
        self.tabs.area.clear()
        with self.tabs.area, ui.row().classes('items-center gap-3 p-4'):
            ui.spinner(size='md')
            ui.label(f'Loading the config schema from incar on {self.host.label}…')
        try:
            schema = await training.schema(self.host)
        except Exception as e:  # show why instead of an empty form
            self.tabs.area.clear()
            with self.tabs.area:
                ui.label(f'Could not load from {self.host.label}: {e}').classes('text-negative')
            return
        live = self.live_features()
        self.tabs.area.clear()
        with self.tabs.area:
            self.tabs.form = ConfigForm(schema, {}, lambda: None, kind='dataset', live_features=live)
        self.tabs.show_form()
        self.tabs.form.load({'video_fps': 30, 'features': copy.deepcopy(live)})  # start with what the core sees

    async def check(self) -> bool:
        name = (self.name.value or '').strip()
        if not training.NAME.fullmatch(name):
            note(self.banner, 'Name the dataset: letters, digits, ".", "_" or "-".', 'bad')
            return False
        if name in self.existing():
            note(self.banner, f'A dataset "{name}" already exists in this workspace.', 'bad')
            return False
        if self.tabs.form is None or not self.tabs.sync():
            return False
        error = await incar_check(self.banner, self.host, self.tabs.form.data, 'dataset')
        if error:
            note(self.banner, error, 'bad')
        return error is None

    async def validate(self) -> None:
        if await self.check():
            note(self.banner, 'incar accepts this dataset config.', 'ok')

    async def create(self) -> None:
        if await self.check() and await self.create_dataset(self.name.value.strip(), self.tabs.form.data):
            self.dialog.close()


class CameraManager:
    """Dialog: the workspace's cameras (workspace_config.json) plus the streams the core shows. Add and remove edit the
    config and reload the workspace (incar has no camera API); hiding only affects this GUI."""

    def __init__(self, context: Callable[[], Awaitable[tuple | None]], streams: Callable[[], list],
                 hidden: set, toggle_hidden: Callable[[str], None], busy: Callable[[], str | None],
                 reload: Callable[[str], Awaitable[bool]]):
        self.get_context, self.streams, self.hidden, self.toggle_hidden = context, streams, hidden, toggle_hidden
        self.busy, self.reload = busy, reload
        self.host = self.workspace = self.schema = None
        self.config: dict = {}
        self.values: dict = {}
        with ui.dialog() as self.dialog, ui.card().classes('w-full max-w-3xl gap-4 p-5'):
            with ui.row().classes('w-full items-center no-wrap'):
                heading('Cameras', 'ph-video-camera')
                ui.space()
                icon_button('ph-x', 'Close', self.dialog.close)
            self.list = ui.column().classes('w-full gap-2')
            ui.separator()
            ui.label('Add a camera to the workspace').classes('section-label')
            with ui.row().classes('w-full no-wrap gap-3'):
                self.name = ui.input('Camera name', placeholder='e.g. wrist_left').classes('grow')
                self.type = ui.select([], label='Type', on_change=self.show_fields).classes('w-48')
            self.fields = ui.column().classes('w-full')
            with ui.row().classes('w-full items-center gap-3'):
                ui.label('Adding or removing reloads the workspace: incar restarts its cameras and deselects '
                         'the dataset and policy.').classes('hint grow')
                ui.button('Add camera', icon='ph-plus-circle', on_click=self.add).props('color=positive')
            self.banner = ui.column().classes('w-full gap-2')

    def config_path(self) -> str:
        return f'{self.workspace}/workspace_config.json'

    async def open(self) -> None:
        context = await self.get_context()
        if context is None:
            return
        self.host, self.workspace = context
        self.banner.clear()
        self.name.set_value('')
        self.dialog.open()
        try:
            self.config = json.loads(await self.host.call('read_text', self.config_path()))
            self.schema = await training.schema(self.host)
        except Exception as e:  # show why instead of an empty dialog
            note(self.banner, f'Could not load the cameras from {self.host.label}: {e}', 'bad')
            return
        self.type.set_options(sorted(self.schema['registries']['camera']), value=self.type.value)
        self.show_fields()
        self.render()

    def render(self) -> None:
        cameras = self.config.get('cameras') or {}
        self.list.clear()
        with self.list:
            for name in sorted(set(cameras) | set(self.streams())):
                with ui.row().classes('demo-item w-full items-center no-wrap gap-3 px-3 py-2'):
                    ui.icon('ph-video-camera' if name in cameras else 'ph-broadcast')
                    with ui.column().classes('gap-0 grow'):
                        ui.label(name)
                        ui.label(cameras[name].get('type', '?') if name in cameras else 'stream from the core') \
                            .classes('hint')
                    hidden = name in self.hidden
                    icon_button('ph-eye-slash' if hidden else 'ph-eye', 'Show in the GUI' if hidden else 'Hide in the GUI',
                                partial(self.hide, name))
                    if name in cameras:
                        icon_button('ph-trash', 'Remove from the workspace', partial(self.remove, name))
            if not cameras:
                ui.label('No cameras in this workspace yet.').classes('hint')

    def hide(self, name: str) -> None:
        self.toggle_hidden(name)
        self.render()

    def show_fields(self) -> None:
        self.values = {}
        self.fields.clear()
        if self.type.value and self.schema:
            with self.fields:
                ConfigForm(self.schema, {}, lambda: None).fields(self.schema['registries']['camera'][self.type.value],
                                                                 self.values, lambda: None)

    async def add(self) -> None:
        name = (self.name.value or '').strip()
        self.banner.clear()
        if not training.NAME.fullmatch(name):
            note(self.banner, 'Name the camera: letters, digits, ".", "_" or "-".', 'bad')
        elif name in (self.config.get('cameras') or {}) or name in self.streams():
            note(self.banner, f'A camera "{name}" already exists.', 'bad')
        elif not self.type.value:
            note(self.banner, 'Choose the camera type.', 'bad')
        elif await self.save({**self.config.get('cameras', {}), name: {'type': self.type.value, **self.values}}):
            self.name.set_value('')
            self.show_fields()

    async def remove(self, name: str) -> None:
        if await confirm(f'Remove camera "{name}" from the workspace? incar stops it and reloads the workspace.',
                         'Remove'):
            await self.save({k: v for k, v in self.config['cameras'].items() if k != name})

    async def save(self, cameras: dict) -> bool:
        """Check the new config with incar, write it and reload the workspace."""
        self.banner.clear()
        if reason := self.busy():
            note(self.banner, f'Not changed: {reason}', 'bad')
            return False
        config = {**self.config, 'cameras': cameras}
        if error := await incar_check(self.banner, self.host, config, 'workspace'):
            note(self.banner, error, 'bad')
            return False
        await self.host.call('write_text', self.config_path(), json.dumps(config, indent=4) + '\n')
        self.config = config
        self.render()
        if await self.reload(self.workspace):
            note(self.banner, 'Saved; incar reloaded the workspace.', 'ok')
        else:
            note(self.banner, 'Saved, but incar did not reload the workspace; reload it in Settings.', 'warn')
        return True


ALL = object()  # ProgressView.selected: all trainings at once
ORIGINS = {False: ('ph-browser', 'started in the GUI'), True: ('ph-terminal-window', 'started outside the GUI')}


class PolicyEditor:
    """Dialog: a trained policy's settings, the policy_config.json of one checkpoint, as form + JSON. Saving checks
    them with incar, keeps the trained original once (.orig) and reloads the policy if the core has it loaded."""

    def __init__(self, context: Callable[[], Awaitable[tuple | None]], loaded: Callable[[], tuple],
                 reload: Callable[[str, str], Awaitable[bool]]):
        self.get_context, self.loaded, self.reload = context, loaded, reload
        self.host = self.workspace = self.schema = self.name = None
        self.type = ''
        self.shown: str | None = None  # the checkpoint whose settings are in the editor
        self.dirty = False
        with ui.dialog().props('maximized persistent') as self.dialog, \
                ui.card().classes('w-full h-full gap-4 no-wrap p-5'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                heading('Policy settings', 'ph-sliders-horizontal')
                self.title = ui.label().classes('text-lg font-medium')
                ui.space()
                icon_button('ph-x', 'Close', self.close)
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                self.checkpoint = ui.select([], label='Checkpoint', on_change=self.load).classes('w-56')
                ui.label('Settings apply when the policy is loaded. Inference values such as n action steps or device '
                         'can change freely; the network architecture and features must stay as trained, or the '
                         'weights no longer load.').classes('hint grow')
                ui.button('Validate', icon='ph-check-circle', on_click=self.validate).props('flat')
                ui.button('Save', icon='ph-floppy-disk', on_click=self.save)
            self.banner = ui.column().classes('w-full gap-2')
            self.tabs = FormJson(self.edited)

    def path(self) -> str:
        return f'{self.workspace}/policies/{self.name}/checkpoints/{self.shown}/model/policy_config.json'

    def edited(self) -> None:
        self.dirty = True

    async def close(self) -> None:
        if not self.dirty or await confirm('Discard the unsaved changes?', 'Discard'):
            self.dialog.close()

    async def open(self, name: str, checkpoint: str | None) -> None:
        context = await self.get_context()
        if context is None:
            return
        self.host, self.workspace = context
        self.name, self.shown, self.dirty = name, None, False
        self.title.set_text(name)
        self.dialog.open()
        try:
            self.schema = await training.schema(self.host)
            checkpoints = await self.host.call('list_names', f'{self.workspace}/policies/{name}/checkpoints')
        except Exception as e:  # show why instead of an empty editor
            self.banner.clear()
            note(self.banner, f'Could not load from {self.host.label}: {e}', 'bad')
            return
        self.checkpoint.set_options(checkpoints, value=None)
        default = next((c for c in (checkpoint, 'last', 'best') if c in checkpoints), checkpoints[-1] if checkpoints else None)
        self.checkpoint.set_value(default)  # loads it

    async def load(self) -> None:
        if self.checkpoint.value == self.shown:
            return  # e.g. switched back below
        if self.dirty and not await confirm('Discard the unsaved changes?', 'Discard'):
            self.checkpoint.set_value(self.shown)  # keep editing the checkpoint the edits belong to
            return
        self.shown, self.dirty = self.checkpoint.value, False
        self.banner.clear()
        self.tabs.area.clear()
        self.tabs.form = None
        if not self.checkpoint.value:
            return
        try:
            config = json.loads(await self.host.call('read_text', self.path()))
        except (RuntimeError, OSError, ValueError) as e:
            note(self.banner, f'Could not read {self.path()}: {e}', 'bad')
            return
        self.type = config.get('type', '')
        fields = self.schema['registries']['policy'].get(self.type)
        if fields is None:
            note(self.banner, f'The policy type "{self.type}" is not installed on {self.host.label}: '
                              'only the JSON view is available.', 'warn')
        with self.tabs.area:
            self.tabs.form = ConfigForm({**self.schema, 'policy': fields or []}, {}, self.edited, kind='policy')
        self.tabs.show_form()
        self.tabs.form.load(config)

    async def check(self) -> bool:
        if self.tabs.form is None or not self.tabs.sync():
            return False
        if self.tabs.form.data.get('type') != self.type:
            note(self.banner, f'The policy type cannot change: the trained weights belong to "{self.type}".', 'bad')
            return False
        error = await incar_check(self.banner, self.host, self.tabs.form.data, 'policy')
        if error:
            note(self.banner, error, 'bad')
        return error is None

    async def validate(self) -> None:
        if await self.check():
            note(self.banner, 'incar accepts these policy settings.', 'ok')

    async def save(self) -> None:
        if not await self.check():
            return
        path, checkpoint = self.path(), self.shown
        if not await self.host.call('exists', f'{path}.orig'):  # the settings it was trained with, kept once
            await self.host.call('write_text', f'{path}.orig', await self.host.call('read_text', path))
        await self.host.call('write_text', path, json.dumps(self.tabs.form.data, indent=4))
        self.dirty = False
        if self.loaded() != (self.name, checkpoint):
            note(self.banner, 'Saved. The settings apply when this checkpoint is loaded.', 'ok')
        elif await self.reload(self.name, checkpoint):
            note(self.banner, 'Saved; incar reloaded the policy with these settings.', 'ok')
        else:
            note(self.banner, 'Saved, but the loaded policy was not reloaded (it is running, or loading failed). '
                              'Load it again to apply.', 'warn')


def run_label(run: training.TrainingRun) -> str:
    p, status = run.progress, run.status()
    if run.running and status == 'Training' and p.total:
        eta = p.eta()
        return f'{p.step / p.total * 100:.0f}%' + (f' · {fmt_duration(eta)} left' if eta else '')
    if run.running and p.task:
        return f'{p.task["name"].lower()} {p.task["done"]}/{p.task["total"]}'
    return status.lower()


def run_state(run: training.TrainingRun) -> str:
    return 'run' if run.running else {'Failed': 'bad', 'Finished': 'ok', 'Stopped early': 'ok'}.get(run.status(), 'idle')


class RunPills:
    """One clickable pill per training (name, progress, state), rebuilt when trainings come or go."""

    def __init__(self, on_click: Callable, active: Callable[[], object] = lambda: None):
        self.on_click, self.active = on_click, active
        self.box = ui.row().classes('gap-2 items-center')
        self.pills: dict[int, tuple] = {}

    def render(self) -> None:
        if [id(r) for r in training.runs] != list(self.pills):
            self.box.clear()
            self.pills.clear()
            with self.box:
                for run in training.runs:
                    with ui.element('div').classes('pill train-pill').props('role=button tabindex=0') as element:
                        ui.element('span').classes('dot')
                        ui.icon('ph-graduation-cap')
                        text = ui.label().classes('truncate max-w-xs')
                        ui.tooltip(f'Show training {run.name}')
                    element.on('click', lambda r=run: self.on_click(r))
                    element.on('keydown.enter', lambda r=run: self.on_click(r))
                    self.pills[id(run)] = (element, text)
        for run in training.runs:
            element, text = self.pills[id(run)]
            set_pill(element, text, f'{run.name} · {run_label(run)}', run_state(run))
            element.classes(add='run-active') if run is self.active() else element.classes(remove='run-active')


class ProgressView:
    """Dialog with the live state of a training (picked from all of them): progress, metrics, loss chart and log.
    Ended trainings of the workspace, also from earlier sessions, can be opened from "Past trainings"."""

    def __init__(self, context: Callable[[], Awaitable[tuple | None]]):
        self.get_context = context
        self.selected: training.TrainingRun | None = None
        self.past: training.TrainingRun | None = None  # the ended training opened from "Past trainings"
        self.run_id = None
        self.seen = 0
        self.chart_size = -1
        with ui.dialog().props('maximized') as self.dialog, ui.card().classes('w-full h-full gap-4 no-wrap p-5'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                heading('Training', 'ph-pulse')
                self.name = ui.label().classes('text-lg font-medium')
                self.state = pill('ph-circle')
                with ui.element('div').classes('pill') as self.origin:  # started in the GUI or outside it
                    self.origin_icon = ui.icon('ph-browser')
                    self.origin_text = ui.label()
                ui.space()
                with ui.button('Past trainings', icon='ph-clock-counter-clockwise').props('flat'):
                    self.history = ui.menu().props('max-height=60vh').classes('min-w-72')
                self.history.on('before-show', self.list_past)
                self.stop_button = ui.button(on_click=self.stop).props('flat color=negative')
                self.dismiss_button = ui.button('Dismiss', icon='ph-x-circle', on_click=self.dismiss).props('flat') \
                    .tooltip('Remove this ended training from the list')
                icon_button('ph-x', 'Close', self.dialog.close)
            with ui.row().classes('w-full items-center gap-2'):
                with ui.element('div').classes('pill train-pill').props('role=button tabindex=0') as self.all_pill:
                    ui.icon('ph-squares-four')
                    ui.label('All trainings')
                self.all_pill.on('click', lambda: self.select(ALL))
                self.all_pill.on('keydown.enter', lambda: self.select(ALL))
                self.selector = RunPills(self.select, lambda: self.selected)
            with ui.column().classes('w-full grow min-h-0 gap-4 no-wrap') as self.overview:  # all trainings at once
                self.overview_rows = ui.element('div').classes('w-full grid grid-cols-1 xl:grid-cols-3 gap-3')
                with ui.column().classes('glass p-3 gap-1 grow min-h-0 w-full'):
                    ui.label('Loss of all trainings').classes('section-label')
                    self.overview_chart = ui.echart(training.combined_chart([], {}), theme='dark') \
                        .classes('w-full grow min-h-64')
            self.overview_ids: list = []
            self.overview_size = None
            self.rows: dict = {}
            self.colors: dict[int, str] = {}  # training -> chart colour, kept while it is listed
            self.empty = ui.label('No training in this session. Start one with "Train new policy" in the Policy '
                                  'panel, or open an earlier one under "Past trainings".') \
                .classes('hint')
            with ui.column().classes('w-full grow min-h-0 gap-4 no-wrap') as self.body:
                with ui.element('div').classes('w-full'):
                    with ui.row().classes('w-full items-baseline justify-between'):
                        self.step_text = ui.label().classes('text-sm')
                        self.percent = ui.label().classes('text-2xl font-semibold tabular-nums')
                    with ui.element('div').classes('train-bar'):
                        self.fill = ui.element('div').classes('train-fill')
                    with ui.row().classes('w-full items-center gap-3 no-wrap mt-2') as self.task_row:
                        ui.spinner(size='sm')
                        self.task_text = ui.label().classes('text-sm whitespace-nowrap')
                        with ui.element('div').classes('train-bar task-bar grow'):
                            self.task_fill = ui.element('div').classes('train-fill').props('data-state=run')
                with ui.element('div').classes('w-full grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3'):
                    self.tiles = {}
                    for key, label in (('loss', 'Training loss'), ('best', 'Best validation'), ('lr', 'Learning rate'),
                                       ('speed', 'Speed'), ('elapsed', 'Elapsed'), ('eta', 'Remaining')):
                        with ui.element('div').classes('stat-tile'):
                            ui.label(label).classes('section-label')
                            self.tiles[key] = (ui.label().classes('stat-value'), ui.label().classes('hint'))
                with ui.element('div').classes('w-full grid grid-cols-1 xl:grid-cols-2 gap-4 grow min-h-0'):
                    with ui.column().classes('glass p-3 gap-1 min-h-0'):
                        ui.label('Loss').classes('section-label')
                        self.chart = ui.echart(training.loss_chart(training.Progress()), theme='dark') \
                            .classes('w-full grow min-h-64')
                        self.plot = ui.image().classes('w-full grow min-h-64 rounded-lg').props('fit=contain')
                    with ui.column().classes('glass p-3 gap-1 min-h-0'):
                        with ui.row().classes('w-full items-center'):
                            ui.label('Log').classes('section-label')
                            ui.space()
                            self.log_path = ui.label().classes('hint truncate')
                        self.log = ui.log(max_lines=3000).classes('w-full grow min-h-64 train-log')
        ui.timer(1.0, self.render)

    def open(self, run: training.TrainingRun | None = None) -> None:
        self.selected = run or self.selected
        self.dialog.open()
        self.render()

    def select(self, run: training.TrainingRun) -> None:
        self.selected = run
        self.render()

    async def list_past(self) -> None:
        self.history.clear()
        context = await self.get_context()
        if context is None:
            return
        host, workspace = context
        try:
            infos = await host.call('runs', workspace)
            policies = await host.call('trained_policies', workspace)
        except (RuntimeError, OSError) as e:
            with self.history:
                ui.item(f'Could not list the trainings on {host.label}: {e}')
            return
        from_gui = {i['policy_name'] for i in infos}
        # (time, info, started outside the GUI); a policy folder no GUI run wrote into was trained outside
        entries = sorted([(i['started'], i, False) for i in infos if not i['alive']] +
                         [(p['checkpoints'][0][1], p, True) for p in policies if p['name'] not in from_gui],
                         key=lambda e: e[0], reverse=True)
        with self.history:
            if not entries:
                ui.item('No ended trainings in this workspace')
            for when, info, outside in entries:
                with ui.menu_item(on_click=partial(self.show_past, host, info, outside)):
                    with ui.item_section().props('avatar'):
                        ui.icon(ORIGINS[outside][0])
                    with ui.item_section():
                        ui.item_label(info['name'])
                        ui.item_label(f'{time.strftime("%d %b %Y, %H:%M", time.localtime(when))} · '
                                      f'{ORIGINS[outside][1]}').props('caption')

    async def show_past(self, host, info: dict, outside: bool) -> None:
        run = None if outside else next(
            (r for r in training.runs if r.host.label == host.label and r.log_path == info['log_path']), None)
        if run is None:  # not from this session
            try:
                run = self.past = await (training.load_policy(host, info) if outside else training.load(host, info))
            except (RuntimeError, OSError) as e:
                ui.notify(f'Could not read the training {info["name"]}: {e}', type='negative')
                return
        self.select(run)

    def render_overview(self, runs: list) -> None:
        """One row per training (state, progress, loss, time left; click opens it) and their losses in one chart."""
        ids = [id(r) for r in runs]
        for key in set(self.colors) - set(ids):
            del self.colors[key]
        for r in runs:  # in order of appearance, never repainted when another training is dismissed
            if id(r) not in self.colors:
                used = set(self.colors.values())
                self.colors[id(r)] = next((c for c in training.SERIES_COLORS if c not in used), '#64748b')
        running = sum(r.running for r in runs)
        self.name.set_text('All trainings')
        set_pill(*self.state, f'{running} of {len(runs)} running', 'run' if running else 'ok')
        if ids != self.overview_ids:
            self.overview_ids, self.overview_size = ids, None
            self.overview_rows.clear()
            self.rows.clear()
            with self.overview_rows:
                for r in runs:
                    with ui.element('div').classes('stat-tile cursor-pointer').props('role=button tabindex=0') as tile:
                        with ui.row().classes('w-full items-center gap-2 no-wrap'):
                            ui.element('span').classes('series-swatch').style(f'background: {self.colors[id(r)]}')
                            ui.label(r.name).classes('font-medium truncate grow')
                            state = pill('ph-circle')
                        with ui.element('div').classes('train-bar my-2'):
                            fill = ui.element('div').classes('train-fill')
                        detail = ui.label().classes('hint')
                        ui.tooltip(f'Show training {r.name}')
                    tile.on('click', lambda r=r: self.select(r))
                    tile.on('keydown.enter', lambda r=r: self.select(r))
                    self.rows[id(r)] = (state, fill, detail)
        for r in runs:
            state, fill, detail = self.rows[id(r)]
            p, status = r.progress, r.status()
            share = min(1.0, p.step / p.total) if p.total else 0.0
            set_pill(*state, status, run_state(r))
            set_style(fill, f'width: {share * 100:.2f}%')
            set_props(fill, f'data-state={"bad" if status == "Failed" else "done" if not r.running else "run"}')
            eta = p.eta() if r.running else None
            detail.set_text(f'Step {p.step:,} of {p.total:,} · {share * 100:.0f}%'
                            + (f' · loss {p.metrics["loss"]}' if 'loss' in p.metrics else '')
                            + (f' · {fmt_duration(eta)} left' if eta else ''))
        size = tuple((len(r.progress.loss), len(r.progress.val)) for r in runs)
        if size != self.overview_size:
            self.overview_size = size
            self.overview_chart.options.clear()
            self.overview_chart.options.update(training.combined_chart(runs, self.colors))
            self.overview_chart.update()

    def dismiss(self) -> None:
        if self.selected in training.runs and not self.selected.running:
            training.runs.remove(self.selected)
            self.selected = None
            self.render()

    async def stop(self) -> None:
        run = self.selected
        if run is None or not run.running:
            return
        if run.stop_requested:
            if await confirm(f'"{run.name}" has not stopped yet. Kill it now? The current step is lost.', 'Kill'):
                await training.stop(run, force=True)
        elif await confirm(f'Stop "{run.name}"? incar saves the loss plot; the last checkpoint stays usable.', 'Stop'):
            await training.stop(run)

    def render(self) -> None:
        if not self.dialog.value:
            return
        runs = training.runs
        combined = self.selected is ALL and len(runs) > 1
        if not combined and self.selected not in runs and self.selected is not self.past:
            self.selected = runs[-1] if runs else None
        run = None if combined else self.selected
        self.selector.render()
        self.all_pill.set_visibility(len(runs) > 1)
        self.all_pill.classes(add='run-active') if combined else self.all_pill.classes(remove='run-active')
        self.overview.set_visibility(combined)
        self.empty.set_visibility(run is None and not combined)
        self.body.set_visibility(run is not None)
        self.origin.set_visibility(run is not None)
        self.stop_button.set_visibility(run is not None and run.running)
        self.dismiss_button.set_visibility(run in runs and not run.running)
        if combined:
            self.render_overview(runs)
        if run is None:
            return
        if self.run_id != id(run):  # a new training: start the log and chart from scratch
            self.run_id, self.seen, self.chart_size = id(run), 0, -1
            self.log.clear()
            icon, text = ORIGINS[run.outside]
            set_props(self.origin_icon, f'name={icon}')
            self.origin_text.set_text(text)
            self.plot.set_source('data:image/svg+xml;base64,' + base64.b64encode(run.plot.encode()).decode()
                                 if run.plot else '')
            self.plot.set_visibility(bool(run.plot))  # incar's own plot: no log to chart
            self.chart.set_visibility(not run.plot)
        p = run.progress
        status = run.status()
        self.name.set_text(run.name)
        set_pill(*self.state, status, {'Finished': 'ok', 'Failed': 'bad', 'Stopped': 'idle', 'Stopped early': 'ok',
                                       'Ended': 'idle'}.get(status, 'run'))
        self.stop_button.set_text('Kill' if run.stop_requested else 'Stop')
        set_props(self.stop_button, f'icon={"ph-prohibit" if run.stop_requested else "ph-stop"}')
        share = min(1.0, p.step / p.total) if p.total else 0.0
        self.percent.set_text(f'{share * 100:.0f}%')
        self.step_text.set_text(f'Step {p.step:,} of {p.total:,}' + (f' · {p.error}' if p.error else ''))
        set_style(self.fill, f'width: {share * 100:.2f}%')
        set_props(self.fill, f'data-state={"bad" if status == "Failed" else "done" if not run.running else "run"}')
        task = p.task if run.running else None
        self.task_row.set_visibility(task is not None)
        if task:  # preprocessing can take longer than the training itself
            self.task_text.set_text(f'{task["name"]} · {task["done"]}/{task["total"]} · {task["left"]} left')
            set_style(self.task_fill, f'width: {task["done"] / max(task["total"], 1) * 100:.1f}%')
        m = p.metrics
        elapsed = (run.ended or time.time()) - run.started
        values = {
            'loss': (m.get('loss', '–'), f'gradient norm {m["grdn"]}' if 'grdn' in m else ''),
            'best': (f'{p.best[1]:.4g}' if p.best else (f'{p.val[-1][1]:.4g}' if p.val else '–'),
                     f'at step {p.best[0]:,}' if p.best else 'validation mse'),
            'lr': (m.get('lr', '–'), ''),
            'speed': (f'{p.rate:.1f} steps/s' if p.rate else '–', f'epoch {m["epch"]}' if 'epch' in m else ''),
            'elapsed': (fmt_duration(elapsed), time.strftime('since %H:%M' if run.running else 'started %d %b, %H:%M',
                                                             time.localtime(run.started))),
            'eta': (fmt_duration(p.eta()) if run.running else '–',
                    time.strftime('done ~%H:%M', time.localtime(time.time() + p.eta())) if run.running and p.eta() else ''),
        }
        for key, (value, sub) in values.items():
            self.tiles[key][0].set_text(str(value))
            self.tiles[key][1].set_text(sub)
        size = (len(p.loss), len(p.val), p.best)
        if size != self.chart_size:
            self.chart_size = size
            self.chart.options.clear()
            self.chart.options.update(training.loss_chart(p))
            self.chart.update()
        self.log_path.set_text(str(run.log_path))
        for seq, text, kind in run.lines:
            if seq > self.seen:
                self.log.push(text, classes=f'log-{kind}')
                self.seen = seq


def header_indicator(show_progress: Callable) -> None:
    """Top bar: one compact pill for all trainings, so many or long-named ones never crowd it; the tooltip lists
    each training, a click opens the progress window."""
    with ui.element('div').classes('pill train-pill').props('role=button tabindex=0') as element:
        ui.element('span').classes('dot')
        ui.icon('ph-graduation-cap')
        text = ui.label().classes('truncate max-w-56')
        tip = ui.tooltip().classes('whitespace-pre-line text-sm')
    def show() -> None:  # several trainings: open them side by side
        show_progress(ALL if len(training.runs) > 1 else None)
    element.on('click', show)
    element.on('keydown.enter', show)

    def render() -> None:
        runs = training.runs
        element.set_visibility(bool(runs))
        if not runs:
            return
        running = sum(r.running for r in runs)
        states = {run_state(r) for r in runs}
        if len(runs) == 1:
            label = f'{runs[0].name} · {run_label(runs[0])}'
        else:
            label = f'{len(runs)} trainings · ' + (f'{running} running' if running else 'done')
        state = 'run' if running else 'bad' if 'bad' in states else 'ok' if 'ok' in states else 'idle'
        set_pill(element, text, label, state)
        tip.set_text('\n'.join(f'{r.name} · {run_label(r)}' for r in runs))
    render()
    ui.timer(1.0, render)
