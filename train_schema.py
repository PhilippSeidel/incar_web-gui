"""Describe or check incar training, dataset and workspace configs. Runs in a separate process: the registries load torch.

    python train_schema.py schema                  -> JSON description of TrainPipelineConfig, DatasetConfig and
                                                      all registered types
    python train_schema.py check FILE|- [KIND]     -> {"ok": true} or {"ok": false, "error": "..."};
                                                      KIND: train (default), dataset, workspace or policy
"""
import contextlib
import dataclasses
import enum
import io
import json
import os
import sys
import types
import typing


def load_registries() -> dict:
    with contextlib.redirect_stdout(io.StringIO()):  # extensions print "registering type ..." while loading
        from incar.extensions.utils import import_native
        import_native()
        from incar.extensions.ai import LRSchedulerConfig, OptimizerConfig, PolicyConfig
        from incar.extensions.cameras import CameraConfig
        from incar.extensions.processing_step import ProcessStep
        registries = {'policy': PolicyConfig, 'optimizer': OptimizerConfig, 'scheduler': LRSchedulerConfig,
                      'step': ProcessStep, 'camera': CameraConfig}
        for registry in registries.values():
            registry.get_known_choices()  # triggers loading the extension entry points
    return registries


def encode(value):
    if isinstance(value, enum.Enum):
        return value.name
    if dataclasses.is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {encode(k): encode(v) for k, v in value.items()}
    return value if value is None or isinstance(value, (bool, int, float, str)) else repr(value)


class Describer:
    def __init__(self, registries: dict):
        self.registry_of = {cls: key for key, cls in registries.items()}
        self.structs: dict[str, list] = {}
        self.enums: dict[str, list] = {}

    def kind(self, tp) -> dict:
        if typing.get_origin(tp) in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(tp) if a is not type(None)]
            inner = self.kind(args[0]) if len(args) == 1 else {'kind': 'any'}
            return {**inner, 'nullable': True}
        origin, args = typing.get_origin(tp) or tp, typing.get_args(tp)
        if tp in (bool, int, float, str):
            return {'kind': tp.__name__}
        if isinstance(tp, type) and issubclass(tp, enum.Enum):
            self.enums[tp.__name__] = [m.name for m in tp]
            return {'kind': 'enum', 'enum': tp.__name__}
        if isinstance(tp, type) and tp in self.registry_of:
            return {'kind': 'choice', 'registry': self.registry_of[tp]}
        if isinstance(tp, type) and dataclasses.is_dataclass(tp):
            if tp.__name__ not in self.structs:
                self.structs[tp.__name__] = []  # placeholder against recursion
                self.structs[tp.__name__] = self.fields(tp)
            return {'kind': 'struct', 'struct': tp.__name__}
        if origin in (list, tuple):
            fixed = origin is tuple and args and args[-1] is not Ellipsis
            item = self.kind(args[0]) if args else {'kind': 'any'}
            return {'kind': 'list', 'item': item, **({'length': len(args)} if fixed else {})}
        if origin is dict:
            return {'kind': 'dict', 'key': self.kind(args[0]) if args else {'kind': 'str'},
                    'value': self.kind(args[1]) if len(args) > 1 else {'kind': 'any'}}
        return {'kind': 'any'}

    def fields(self, cls) -> list[dict]:
        hints = typing.get_type_hints(cls)
        result = []
        for f in dataclasses.fields(cls):
            if not f.init:
                continue
            if f.default is not dataclasses.MISSING:
                default, required = encode(f.default), False
            elif f.default_factory is not dataclasses.MISSING:
                factory = f.default_factory  # some extensions pass a plain value instead of a callable
                default, required = encode(factory() if callable(factory) else factory), False
            else:
                default, required = None, True
            result.append({'name': f.name, **self.kind(hints[f.name]), 'default': default, 'required': required})
        return result


def schema() -> dict:
    registries = load_registries()
    from incar.config.dataset_config import DatasetConfig
    from incar.config.training_pipeline import TrainPipelineConfig
    d = Describer(registries)
    out = {'train': d.fields(TrainPipelineConfig), 'dataset': d.fields(DatasetConfig),
           'registries': {key: {name: d.fields(cls) for name, cls in reg.get_known_choices().items()}
                          for key, reg in registries.items()}}
    return {**out, 'structs': d.structs, 'enums': d.enums}


def check(path: str, kind: str = 'train') -> dict:
    registries = load_registries()
    import draccus
    import tempfile
    from incar.config.dataset_config import DatasetConfig
    from incar.config.training_pipeline import TrainPipelineConfig
    from incar.config.workspace_config import WorkspaceConfig
    if path == '-':  # config on stdin: works the same over SSH
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            f.write(sys.stdin.read())
        try:
            return check(f.name, kind)
        finally:
            os.unlink(f.name)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            config_class = {'dataset': DatasetConfig, 'workspace': WorkspaceConfig,
                            'policy': registries['policy']}.get(kind, TrainPipelineConfig)
            cfg = draccus.parse(config_class=config_class, config_path=path, args=[])
            if hasattr(cfg, 'validate'):  # WorkspaceConfig and PolicyConfig have none
                cfg.validate()
    except BaseException as e:  # draccus reports bad configs via SystemExit as well
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    return {'ok': True}


if __name__ == '__main__':
    result = schema() if sys.argv[1] == 'schema' else check(*sys.argv[2:4])
    sys.stdout.write('\n@@JSON@@' + json.dumps(result))  # marker: extensions may still print to stdout
