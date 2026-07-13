# verl + SGLang GRPO on Intel Gaudi 2

Run verl GRPO on Intel Gaudi 2 (HL-225) with SGLang as the rollout engine, and have
it **actually learn** — not merely run crash-free.

- Model: `Qwen3-4B-Base` (`/scratch/rarya124/models/Qwen3-4B-Base`)
- Data: DAPO-Math-17k, `gaudi/data/dapo_math/{train,val}.parquet` (13,116 / 1,000 rows)
- Algorithm: plain GRPO (`verl.trainer.main_ppo`, `algorithm.adv_estimator=grpo`)
- Rollout: SGLang. **vLLM is not a goal.**

## Success bar

```
critic/score/mean > -1.0, with varied rewards across the group
  => nonzero critic/advantages
  => nonzero actor/pg_loss
```

GRPO advantage = reward − group mean. If every sample in a group scores identically,
the advantage is 0 and `actor/pg_loss` is exactly 0.0 — **the model learns nothing**.
A run that completes steps while logging `critic/score/mean: -1.0` has not learned;
it has only failed slowly.

## Layout

```
sglang/               upstream SGLang (this repo's fork base)
verl/                 upstream verl, vendored at 30119a25 (v0.8.0-176-g30119a25)
gaudi/
  verl_hpu_plugin/    verl external plugin: HPU platform + FSDP engine registration
  recipe/grpo_qwen3_4b_gaudi/
    run_grpo_sglang_hpu.sh   the recipe
    grpo_gaudi.sbatch        Slurm wrapper (4 HPUs, apptainer)
  data/dapo_math/     the verified dataset
```

`verl/` is vendored (its `.git` was dropped), so local edits to it are tracked here.
Its upstream is `https://github.com/verl-project/verl.git` at commit `30119a25`.

## Why the plugin, and not a verl fork

verl exposes two extension points that make a fork unnecessary:

- `PlatformRegistry` (`verl/plugin/platform/`) — `device.py` resolves every device
  call through `get_platform()`, so a `PlatformBase` subclass is enough to teach verl
  what an HPU is.
- `EngineRegistry.register(...)` is an ordinary classmethod decorator, so verl's own
  FSDP engines can be re-registered under `device='hpu'` from outside. The engines are
  already device-agnostic; they were simply never declared for hpu.

Both are activated by env var, so the verl source tree stays pristine:

```bash
VERL_USE_EXTERNAL_MODULES=verl_hpu_plugin
VERL_PLATFORM=hpu
```

Without the engine registration, `main_ppo` fails with
`ValueError: No engine registered for device='hpu' ... backend='fsdp'`.

## Hard-won constraints — change these only with evidence

Each of these cost a job (or several) to learn.

- **One process per Habana module.** A colocated / hybrid engine (rollout server
  sharing a module with an FSDP worker) is *impossible* — it dies with
  `synStatus=8 [Device not found]`, deterministically. Only the disaggregated layout
  works: FSDP on modules 0,1 and rollout on 2,3 via `HPU_ROLLOUT_MODULE_OFFSET=2`.
- **HPU hates dynamic shapes.** Every distinct tensor shape compiles a new Habana
  graph. `use_dynamic_bsz=True` packs a different token count per micro-batch and
  eventually dies with `Graph duplication failed. synStatus=26`. Use fixed
  `ppo_micro_batch_size_per_gpu` / `log_prob_micro_batch_size_per_gpu`.
- **Ray needs `ray_kwargs.ray_init.num_cpus=72`.** The cgroup grants 72 CPUs but the
  container sees the host's 152, mis-sizes its pool, and hangs at bootstrap.
- **`SGLANG_HPU_PREFILL_BUCKET_MAX` must be ≥ `max_prompt_length`.** Otherwise a short
  prompt pads up to an oversized FusedSDPA graph and overruns the KV cache. Real DAPO
  prompts measure 93–251 tokens.
- **`data.prompt_key=source_prompt`, not the default `prompt`.** The parquet has both:
  `prompt` is raw text, `source_prompt` is the chat-formatted column, and only it
  carries the `Answer: $Answer` instruction that the `dapo` reward manager requires.
  Point verl at `prompt` and every sample scores −1.0 for formatting alone.
- **No IPC on Gaudi.** Synapse has no `cudaIpc*` equivalent, so weight transfer to a
  disaggregated rollout engine must go through the filesystem
  (`update_weights_from_disk`). `PlatformHPU.is_ipc_supported()` returns False.

## Status

Not yet run end-to-end. The open question inherited from the previous attempt is
whether **SGLang on HPU can generate coherent text at all** — every prior run produced
token salad, scoring a uniform −1.0 and yielding `pg_loss = 0.0`. That question has
never been answered in isolation, and it should be answered before trusting any
training metric: load real weights from disk (`load_format=auto`), TP=1, one prompt,
greedy, no verl / FSDP / Ray. Note that upstream SGLang has no HPU backend at all —
`HabanaAI/sglang-fork` (SGLang 0.4.9) is reference material for what that backend has
to do, not a dependency.
