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
- Colab debug script: `examples/colab_lora_debug.py`
- Chat-template teaching note: `examples/chat_template_tokenize_collate_guide.html`

Install:

```bash
pip install -e ".[finetune]"
```

Run local QLoRA training:

```bash
python -m mmit2.training --config configs/local_qlora.yaml
```
