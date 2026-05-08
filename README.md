# mmit2

Minimal multimodal instruction tuning package for the initial release.

Current training methods:

- `lora`
- `qlora`
- `dora`
- `freeze`
- `l2t`

Current training data path:

- Hugging Face datasets only
- built-in VQA-style dataset support for:
  - `lmms-lab/VQAv2`
  - `lmms-lab/textvqa`
  - `lmms-lab/VizWiz-VQA`

Current evaluation support:

- `VQAv2`
- `POPE`
- `MME`

Useful entry points:

- Training configs: `configs/`
- Training CLI: `python -m mmit2.training --config configs/local_qlora.yaml`
- Debug CLI: `python -m mmit2.debug --config configs/colab_lora_debug.yaml`
- Smoke CLI: `python -m mmit2.smoke --suite quick`
- Colab debug wrapper: `examples/colab_lora_debug.py`
- Smoke matrix wrapper: `examples/colab_smoke_matrix.py`
- Chat-template teaching note: `examples/chat_template_tokenize_collate_guide.html`

Install:

```bash
pip install -e ".[finetune]"
```

Run local QLoRA training:

```bash
python -m mmit2.training --config configs/local_qlora.yaml
```

Run the package-level debug entry point:

```bash
python -m mmit2.debug --config configs/colab_lora_debug.yaml
```

Run the package-level smoke matrix:

```bash
python -m mmit2.smoke --suite quick
```

Compatibility note:

- `examples/colab_lora_debug.py` and `examples/colab_smoke_matrix.py` are still kept for Colab notebooks, teaching notes, and existing test flows.
- New work should prefer the package-level commands above so users do not have to treat `examples/` as the main interface.
