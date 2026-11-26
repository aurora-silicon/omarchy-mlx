# Online attention kernels

The omarchy Metal backend ships four online-softmax SDPA paths for Apple GPUs.
This page records when each one dispatches, its defaults, and the environment
overrides. Every route is bitwise identical to the composed fallback at every
key length — the engagement windows move only the wall clock, never a token.

All online paths require bf16 attention, causal masking, and an Apple GPU
device reporting subgroup size 32 with the basic, arithmetic, and shuffle
subgroup features. The composed route serves everything else, including f16.

## Online decode — `MLX_OMARCHY_SDPA_ONLINE`

One-dispatch online-softmax decode for head_dim 128 (commit `5e26721a6a`).

| condition | value |
|---|---|
| query length | exactly 1 (decode) |
| head_dim | 128 |
| workgroup | 16 or 32 rows per subgroup |
| default | off (the env is unset) |

Set `MLX_OMARCHY_SDPA_ONLINE=16` or `=32` to engage; unset or any other value
keeps the composed route. `MLX_OMARCHY_SDPA_DECODE_NATIVE=0` disables the
native decode arm outright. Device gates: work-group invocations and size at
least `SG × 32`, shared memory at least `SG × 130` floats. Measured on the
16-inch M1 Max, 4B decode: 84.45 → 91.77 tok/s with the kernel engaged.

## Online prefill — `MLX_OMARCHY_SDPA_PREFILL_ONLINE`

Prefill counterpart at head_dim 128, fixed 8 rows per subgroup (issue #377
lineage, SUP-118).

| condition | value |
|---|---|
| causal, query length | > 16 |
| key length | ≤ 2048 (hard cap) |
| workgroup | 8 rows per subgroup |
| default | off |

Set `MLX_OMARCHY_SDPA_PREFILL_ONLINE=1` to engage. The 2048-key cap is
measured, not arbitrary: on the 16-inch M1 Max the online prefill loses
9.49 % of prefill throughput at ctx2048 and is noise at ctx512 (five paired
ABBA blocks, identical token ids — SUP-118). Shared-memory demand at ctx2048
is 33.6 MB against the 48 MB SLC; at ctx4096 it would be 67 MB and spill.
Do not enable this env for contexts at or near 2048 on current silicon.

## Split-K decode rows — bf16, query length 2–16

The default decode route for bf16 where its measured window holds. One
pass-1/pass-2 pair per row; `MLX_OMARCHY_SDPA_SPLIT_ROWS=0` keeps the
composed route.

| query width | key window |
|---|---|
| 64 | 256 – 2048 |
| 128 | 1 – 7168 |
| 256 | 12 – 128 |

Beyond the window the composed route wins (the width-256 arm loses 33–46 %
in the 448–544 contract regime; the single-workgroup walk loses to parallel
GEMV tiles once the key stream dominates). HD=256 rows engage under the
#997/#1001 calibration: cap 1024 keys on G14C, 2304 on G13C at HD=128;
G13C HD=256 stays uncalibrated and off.

## Chunked composed route

Long-context calls beyond the online caps compose from chunked passes.
`MLX_OMARCHY_SDPA_CHUNK_MAX_BYTES` (bytes, default 1 GiB) caps the per-call
chunk scratch; a call over the cap splits into chunks.

## Override summary

| env | values | effect |
|---|---|---|
| `MLX_OMARCHY_SDPA_ONLINE` | unset, 16, 32 | online decode workgroup; off unless 16/32 |
| `MLX_OMARCHY_SDPA_DECODE_NATIVE` | 0 | disable the native decode arm |
| `MLX_OMARCHY_SDPA_PREFILL_ONLINE` | set = on | online prefill, ≤ 2048 keys |
| `MLX_OMARCHY_SDPA_SPLIT_ROWS` | 0 | composed route instead of split-K rows |
| `MLX_OMARCHY_SDPA_DECODE_BATCH` | 0 | decode batch gate back to batch == 1 |
| `MLX_OMARCHY_SDPA_DECODE_ROWS` | 0 | disable decode rows arm |
| `MLX_OMARCHY_SDPA_BF16_FAST` | set = on | bf16 rides the native shapes |
| `MLX_OMARCHY_SDPA_CHUNK_MAX_BYTES` | bytes | chunk scratch cap (default 1 GiB) |
