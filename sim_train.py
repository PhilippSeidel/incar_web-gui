"""Stand-in for `incar train` that only prints a realistic log (almost no CPU): `sim_train.py train --config_path F`.

The GUI runs this instead of incar when TRAIN_SIMULATION=1, to try the training views without a real training.
"""
import json
import math
import os
import random
import sys
import time

config = json.load(open(sys.argv[sys.argv.index('--config_path') + 1]))
steps, log_freq = config.get('steps', 100_000), config.get('log_freq', 200)
validate_freq, save_freq = config.get('validate_freq', 0), config.get('save_freq', 20_000)
duration = float(os.getenv('SIM_SECONDS', '30'))  # length of the simulated training, preprocessing excluded
step_time = duration / max(steps, 1)
bar_pause = min(0.3, duration / 60)


def log(message: str) -> None:
    print(f'INFO {time.strftime("%Y-%m-%d %H:%M:%S")} {"_train.py:361":>15} {message}', flush=True)


try:
    log(f"{{'batch_size': {config.get('batch_size', 8)}, 'steps': {steps}, 'simulated': True}}")
    log('Creating datasets')
    for name in ('Sample dt', 'Downsample video'):
        for i in range(1, 11):
            sys.stdout.write(f'\r{name}: {i * 10:3d}%|{"█" * i}{" " * (10 - i)}| {i}/10 [00:{i:02d}<00:{10 - i:02d}, 1.00it/s]')
            sys.stdout.flush()
            time.sleep(bar_pause)
        print(flush=True)
    log('Creating policy')
    log(f'Output dir: {config.get("model_save_name")}')
    log('Start offline training on a fixed dataset')
    best = math.inf
    for step in range(1, steps + 1):
        if step % max(1, steps // 300) == 0:
            time.sleep(step_time * max(1, steps // 300))
        loss = 1.5 * math.exp(-step / (steps / 4)) + 0.05 + random.uniform(-0.02, 0.02)
        if validate_freq and step % validate_freq == 0:
            mse = loss * random.uniform(1.1, 1.4)
            log(f'Ran validation with an action mse of {mse}')
            if mse < best:
                best = mse
                log(f'Saving new best model at step {step}')
        if step % log_freq == 0:
            log(f'step:{step if step < 1000 else f"{step // 1000}K"} smpl:{step * 8} ep:{step // 50} '
                f'epch:{step / 400:.2f} loss:{loss:.3f} grdn:{random.uniform(0.5, 2):.3f} lr:{1e-4:.1e} '
                f'updt_s:0.050 data_s:0.001')
        if step % save_freq == 0 or step == steps:
            log(f'Checkpoint policy after step {step}')
    log('End of training')
except KeyboardInterrupt:
    log('Training interrupted by user')
