# MoReS Memory Probe

Run these scripts on JarvisLabs A100 to measure whether MoReS fits on a 32 GB RTX 5090.

Each run creates a unique generated config under `experiment_setup/mores_memory_probe/generated/`, trains with `per_device_batch_size=1`, records `nvidia-smi` memory once per second, and prints the peak memory at the end.

If you need a Hugging Face token, put it in `.hf_token` at the repository root on the remote machine. The probe scripts automatically pass it to `python -m vlmintune.training`.

```bash
printf "%s" "$HF_TOKEN" > .hf_token
chmod 600 .hf_token
```

## One-Command 5090 Check

For the first pass, run every built-in dataset with a small sample count for both Qwen and LLaVA:

```bash
bash experiment_setup/mores_memory_probe/run_5090_check.sh
```

By default this runs:

- Qwen2.5-VL-3B MoReS and LLaVA-1.5-7B MoReS
- TextVQA, VizWiz, VQAv2, and GQA
- 50 training samples per dataset
- max_length 1024

This is enough for a fast first 5090 fit check across dataset types.

To run only TextVQA with a larger sample count:

```bash
DATASETS="textvqa" SAMPLE_SIZES="100 1000" bash experiment_setup/mores_memory_probe/run_5090_check.sh
```

To keep all datasets but use 100 samples each:

```bash
SAMPLE_SIZES="100" bash experiment_setup/mores_memory_probe/run_5090_check.sh
```

To also test `max_length: 1536`:

```bash
MAX_LENGTHS="1024 1536" bash experiment_setup/mores_memory_probe/run_5090_check.sh
```

## Wider Matrix

If you want explicit per-model matrix scripts, start with:

```bash
bash experiment_setup/mores_memory_probe/run_qwen_all_datasets_100_len1024.sh
bash experiment_setup/mores_memory_probe/run_llava_all_datasets_100_len1024.sh
```

If those are stable, run the 1000-sample probes:

```bash
bash experiment_setup/mores_memory_probe/run_qwen_all_datasets_1000_len1024.sh
bash experiment_setup/mores_memory_probe/run_llava_all_datasets_1000_len1024.sh
```

Then test longer sequence length:

```bash
bash experiment_setup/mores_memory_probe/run_qwen_all_datasets_1000_len1536.sh
bash experiment_setup/mores_memory_probe/run_llava_all_datasets_1000_len1536.sh
```

Full TextVQA probes are available after the 1000-sample probes look safe:

```bash
bash experiment_setup/mores_memory_probe/run_qwen_textvqa_full_len1024.sh
bash experiment_setup/mores_memory_probe/run_llava_textvqa_full_len1024.sh
```

## One-Off Command

You can also run one combination directly:

```bash
bash experiment_setup/mores_memory_probe/run_probe.sh qwen textvqa 100 1024
bash experiment_setup/mores_memory_probe/run_probe.sh llava vizwiz 1000 1536
```

Arguments:

```text
run_probe.sh <qwen|llava> <textvqa|vizwiz|vqav2|gqa> <max_samples|0> <max_length>
```

Use `max_samples=0` for the full training split.

## What To Send Back

For each run, send:

- The printed `peak_memory_used_gib`
- The `First batch shapes` line from `experiments/<name>/train/run.log`
- `experiments/<name>/train/train_summary.json`, if the run completed
- Any OOM traceback
- Whether the A100 is 40 GB or 80 GB

5090 rule of thumb:

```text
<= 27 GB peak: likely safe
27-30 GB: possible but risky
30-32 GB: high OOM risk
> 32 GB: not suitable without reducing max_length or changing method
```
