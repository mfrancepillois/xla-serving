# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Samples host and accelerator resource usage of a server process tree.

Started by `run_benchmark.sh` next to every load-generator run and stopped with
SIGINT/SIGTERM. The summary JSON is rewritten atomically on every sample so a
kill at any point leaves a valid file:

  {
    "pid": ..., "samples": N, "wall_s": ..., "cpu_s": ...,      # user+sys of
    "avg_cpu_cores": ...,                                        # the tree
    "peak_rss_mb": ..., "avg_rss_mb": ...,                       # see below
    "memory_metric": "rss" | "pss",
    "num_threads_max": ..., "num_processes_max": ...,
    "gpu": {"index": ..., "index_source": ..., "util_avg_pct": ...,
            "util_max_pct": ..., "mem_used_max_mb": ...}
  }

CPU and memory are summed over the root process and all its descendants, so a
multi-process server (the `py-mp*` arms: device owner + N frontends) is
charged for all of them. For such trees memory is the proportional set size
(PSS) rather than RSS, otherwise pages shared between the processes (the
shared-memory rings) would be counted once per process; the key names stay
`*_rss_mb` for report compatibility and `memory_metric` says which it was.

GPU numbers come from `nvidia-smi` (`--accelerator=cuda`) or `rocm-smi`
(`--accelerator=rocm`), when the tool is present.
TPU utilisation is not sampled (there is no stable CLI for it); HBM usage can be
read from the server logs instead.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence

from absl import app
from absl import flags
from absl import logging
import psutil


@dataclasses.dataclass(frozen=True)
class _Flags:
  """Command-line flags; only registered when run as a binary."""

  pid: flags.FlagHolder[int | None]
  output: flags.FlagHolder[str | None]
  interval_s: flags.FlagHolder[float]
  accelerator: flags.FlagHolder[str]
  gpu_index: flags.FlagHolder[int]


def _define_flags() -> _Flags:
  """Registers flags lazily so tests can import `monitor` alongside other tools."""
  return _Flags(
      pid=flags.DEFINE_integer('pid', None, 'Root PID of the server.'),
      output=flags.DEFINE_string('output', None, 'Summary JSON path.'),
      interval_s=flags.DEFINE_float('interval_s', 0.5, 'Sampling interval.'),
      accelerator=flags.DEFINE_enum(
          'accelerator',
          'none',
          ['none', 'cpu', 'cuda', 'rocm', 'tpu'],
          'Accelerator to sample ("cpu" is an alias of "none" so the value of'
          ' run_benchmark.sh --platform can be forwarded verbatim).',
      ),
      gpu_index=flags.DEFINE_integer(
          'gpu_index',
          -1,
          'GPU to sample. -1 resolves it from the server process tree via'
          ' amd-smi (ROCm only, since HIP and rocm-smi device orders differ);'
          ' CUDA falls back to index 0.',
      ),
  )


def _tree(root: psutil.Process) -> list[psutil.Process]:
  try:
    return [root] + root.children(recursive=True)
  except psutil.NoSuchProcess:
    return []


class _TreeSampler:
  """Accumulates CPU deltas per `(pid, create_time)` across a process tree.

  Tracking deltas per process incarnation ensures that child processes exiting
  mid-run never subtract their lifetime CPU from subsequent samples (which
  could otherwise undercount or make `cpu_s` negative).
  """

  def __init__(
      self,
      root: Any,
      tree_fn: Callable[[Any], Sequence[Any]] = _tree,
  ):
    self._root = root
    self._tree_fn = tree_fn
    self._prev_cpu: dict[tuple[int, float], float] = {}
    self.cpu_s = 0.0
    self.samples = 0
    self.peak_mem = 0
    self.mem_sum = 0
    self.threads_max = 0
    self.procs_max = 0
    self.metric = 'rss'
    for p in self._tree_fn(self._root):
      try:
        with p.oneshot():
          key = (p.pid, p.create_time())
          t = p.cpu_times()
          self._prev_cpu[key] = t.user + t.system
      except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

  def sample(self) -> bool:
    """Takes one sample of the process tree; returns False if root exited."""
    procs = self._tree_fn(self._root)
    if not procs:
      return False
    mem = 0
    threads = 0
    metric = 'pss' if len(procs) > 1 else 'rss'
    for p in procs:
      try:
        with p.oneshot():
          key = (p.pid, p.create_time())
          t = p.cpu_times()
          cur_cpu = t.user + t.system
          prev_cpu = self._prev_cpu.get(key, 0.0)
          if cur_cpu > prev_cpu:
            self.cpu_s += cur_cpu - prev_cpu
          self._prev_cpu[key] = cur_cpu
          threads += p.num_threads()
          if metric == 'pss':
            try:
              mem += p.memory_full_info().pss
            except (psutil.AccessDenied, AttributeError):
              mem += p.memory_info().rss
              metric = 'rss'
          else:
            mem += p.memory_info().rss
      except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
    self.samples += 1
    self.peak_mem = max(self.peak_mem, mem)
    self.mem_sum += mem
    self.threads_max = max(self.threads_max, threads)
    self.procs_max = max(self.procs_max, len(procs))
    self.metric = metric
    return True


def _nvidia_gpu_sample(index: int) -> tuple[float, float] | None:
  """Queries `(gpu_util_pct, mem_used_mb)` for GPU `index` via `nvidia-smi`."""
  if not shutil.which('nvidia-smi'):
    return None
  try:
    out = subprocess.run(
        [
            'nvidia-smi',
            f'--id={index}',
            '--query-gpu=utilization.gpu,memory.used',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout.strip()
    util, mem = out.split(',')
    return float(util), float(mem)
  except (subprocess.SubprocessError, ValueError):
    return None


def _rocm_all_cards() -> dict[int, tuple[float, float]]:
  """Reads `{card: (gpu_util_pct, mem_used_mb)}` for every GPU via `rocm-smi`.

  `rocm-smi` has no per-device query flag, so it reports every GPU at once:

      device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)
      card0,0,270566162432,297766912

  Columns are located by header text rather than position, since their order
  and exact wording have moved between ROCm releases. Bytes are converted to
  MiB so the value means the same thing as the `nvidia-smi` one.
  """
  if not shutil.which('rocm-smi'):
    return {}
  try:
    out = subprocess.run(
        ['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--csv'],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip().splitlines()
  except subprocess.SubprocessError:
    return {}
  # Some builds wrap the CSV in banner lines, so find the header rather than
  # assuming it is the first line.
  util_col = mem_col = None
  for i, line in enumerate(out):
    cols = [c.strip().lower() for c in line.split(',')]
    util_col = next((j for j, c in enumerate(cols) if 'use (%)' in c), None)
    mem_col = next((j for j, c in enumerate(cols) if 'used memory' in c), None)
    if util_col is not None and mem_col is not None:
      out = out[i + 1 :]
      break
  else:
    return {}
  cards: dict[int, tuple[float, float]] = {}
  for line in out:
    row = [c.strip() for c in line.split(',')]
    if not row[0].startswith('card') or not row[0][4:].isdigit():
      continue
    if len(row) <= max(util_col, mem_col):
      continue
    try:
      cards[int(row[0][4:])] = (
          float(row[util_col]),
          float(row[mem_col]) / (1 << 20),
      )
    except ValueError:
      continue
  return cards


def _rocm_gpu_sample(index: int) -> tuple[float, float] | None:
  """Queries `(gpu_util_pct, mem_used_mb)` for GPU `index` via `rocm-smi`."""
  return _rocm_all_cards().get(index)


_GPU_SAMPLERS = {'cuda': _nvidia_gpu_sample, 'rocm': _rocm_gpu_sample}


@functools.lru_cache(maxsize=None)
def _warn_once(message: str) -> None:
  logging.warning('%s', message)


def _gpu_resolution_failed(reason: str) -> None:
  """Logs why GPU resolution failed, once per distinct reason.

  Resolution is retried every sample, so this must not log per sample; but a
  silent failure leaves an empty `gpu` block in the summary with no way to
  tell whether the GPU was idle or never found.
  """
  _warn_once(f'Cannot resolve GPU to sample: {reason}')


def _rocm_gpu_for_pids(pids: set[int]) -> int | None:
  """Finds the AMD GPU index holding an allocation for one of `pids`.

  HIP device ordering and `rocm-smi`'s `cardN` ordering do not agree: with
  `HIP_VISIBLE_DEVICES=0` a server has been observed on `card3`. Sampling a
  fixed index therefore reports an unrelated, idle GPU. `amd-smi` maps
  processes to devices directly, which removes the guess; its GPU index does
  agree with `rocm-smi`'s card index.

  Returns `None` while no allocation is visible yet, so the caller can retry.
  """
  if not pids:
    _gpu_resolution_failed('monitored process tree is empty')
    return None
  if not shutil.which('amd-smi'):
    _gpu_resolution_failed('amd-smi not found on PATH')
    return None
  try:
    proc = subprocess.run(
        ['amd-smi', 'process', '--json'],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
  except subprocess.SubprocessError as e:
    _gpu_resolution_failed(f'amd-smi process failed: {e}')
    return None
  try:
    data = json.loads(proc.stdout)
  except json.JSONDecodeError:
    _gpu_resolution_failed(f'amd-smi JSON unparseable: {proc.stdout[:200]!r}')
    return None
  seen: dict[Any, list[int]] = {}
  for entry in data if isinstance(data, list) else []:
    for p in entry.get('process_list', []):
      # `process_info` is the string "No running processes detected" when the
      # device is idle, and a dict otherwise.
      info = p.get('process_info')
      if not isinstance(info, dict):
        continue
      seen.setdefault(entry.get('gpu'), []).append(info.get('pid'))
      if info.get('pid') in pids and isinstance(entry.get('gpu'), int):
        return entry['gpu']
  _gpu_resolution_failed(
      f'no amd-smi GPU holds any of pids {sorted(pids)}; GPUs with '
      f'processes: {seen or "none"}'
  )
  return None


# A card must hold this many times the runner-up's VRAM to be judged "the one
# in use". On an idle 8-GPU host every card sits within noise of the others,
# and picking the nominal maximum would be arbitrary.
_BUSIEST_CARD_RATIO = 2.0


def _rocm_busiest_card() -> int | None:
  """Picks the GPU holding conspicuously more VRAM than any other.

  A fallback for when `amd-smi` cannot map PIDs to devices. The benchmark pins
  the server to one GPU, so on a host where the other cards are idle the one
  in use stands out by orders of magnitude. Returns `None` when no card stands
  out, rather than guessing -- a shared host where someone else's job is the
  largest consumer would otherwise be silently mis-attributed.
  """
  cards = _rocm_all_cards()
  if not cards:
    return None
  if len(cards) == 1:
    return next(iter(cards))
  ranked = sorted(cards, key=lambda i: cards[i][1], reverse=True)
  top, runner_up = cards[ranked[0]][1], cards[ranked[1]][1]
  if top < _BUSIEST_CARD_RATIO * runner_up:
    _gpu_resolution_failed(
        f'no GPU stands out by VRAM (top card{ranked[0]}={top:.0f} MiB vs '
        f'card{ranked[1]}={runner_up:.0f} MiB)'
    )
    return None
  return ranked[0]


def _resolve_rocm_gpu(pids: set[int]) -> tuple[int | None, str | None]:
  """Returns `(gpu_index, how_it_was_resolved)`."""
  gpu = _rocm_gpu_for_pids(pids)
  if gpu is not None:
    return gpu, 'amd-smi-pid'
  gpu = _rocm_busiest_card()
  if gpu is not None:
    _warn_once(f'Falling back to VRAM-based GPU detection: card{gpu}')
    return gpu, 'max-vram'
  return None, None


def _cpu_affinity(proc: Any) -> list[int] | None:
  if not hasattr(proc, 'cpu_affinity'):
    return None
  try:
    return sorted(proc.cpu_affinity())
  except (psutil.Error, OSError):
    return None


def main(argv: Sequence[str], f: _Flags) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  if f.pid.value is None or not f.output.value:
    raise app.UsageError('--pid and --output are required')
  if f.interval_s.value <= 0:
    raise app.UsageError('--interval_s must be > 0')
  try:
    root = psutil.Process(f.pid.value)
  except psutil.NoSuchProcess:
    logging.warning('Process %d already exited; stopping monitor', f.pid.value)
    sys.exit(0)
  out_path = os.path.abspath(f.output.value)
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  stop_event = threading.Event()

  def _on_signal(signum, frame):  # pylint: disable=unused-argument
    stop_event.set()

  signal.signal(signal.SIGINT, _on_signal)
  signal.signal(signal.SIGTERM, _on_signal)

  sampler = _TreeSampler(root)
  t0 = time.monotonic()
  gpu_utils: list[float] = []
  gpu_mem_max = 0.0
  summary: dict[str, Any] = {}

  # `-1` means auto-detect, which only ROCm supports.
  gpu_index: int | None = None
  gpu_index_source: str | None = None
  if f.gpu_index.value >= 0:
    gpu_index, gpu_index_source = f.gpu_index.value, 'flag'
  elif f.accelerator.value != 'rocm':
    gpu_index, gpu_index_source = 0, 'default'

  while not stop_event.is_set():
    stop_event.wait(f.interval_s.value)
    if stop_event.is_set() and sampler.samples > 0:
      break
    if not sampler.sample():
      logging.warning('Process %d exited; stopping monitor', f.pid.value)
      break
    sampler_fn = _GPU_SAMPLERS.get(f.accelerator.value)
    if sampler_fn is not None:
      if gpu_index is None:
        # Resolution needs a visible allocation, which a server may not have
        # made yet, so keep retrying until it succeeds.
        gpu_index, gpu_index_source = _resolve_rocm_gpu(
            {p.pid for p in _tree(root)}
        )
      if gpu_index is not None:
        g = sampler_fn(gpu_index)
        if g is not None:
          gpu_utils.append(g[0])
          gpu_mem_max = max(gpu_mem_max, g[1])
    wall = time.monotonic() - t0
    summary = {
        'pid': f.pid.value,
        'samples': sampler.samples,
        'wall_s': wall,
        'cpu_s': sampler.cpu_s,
        'avg_cpu_cores': sampler.cpu_s / wall if wall > 0 else 0.0,
        'peak_rss_mb': sampler.peak_mem / 2**20,
        'avg_rss_mb': sampler.mem_sum / sampler.samples / 2**20,
        'memory_metric': sampler.metric,
        'num_threads_max': sampler.threads_max,
        'num_processes_max': sampler.procs_max,
        'server_cpu_affinity': _cpu_affinity(root),
    }
    if gpu_utils:
      summary['gpu'] = {
          'index': gpu_index,
          'index_source': gpu_index_source,
          'util_avg_pct': sum(gpu_utils) / len(gpu_utils),
          'util_max_pct': max(gpu_utils),
          'mem_used_max_mb': gpu_mem_max,
      }
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as out_f:
      json.dump(summary, out_f, indent=2)
    os.replace(tmp, out_path)

  logging.info('Monitor summary: %s', json.dumps(summary))
  sys.exit(0)


if __name__ == '__main__':
  app.run(functools.partial(main, f=_define_flags()))
