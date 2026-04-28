# DevMentor LoRA → Ollama (Option A)

Pipeline: **HF adapter** → **merged HF model** → **GGUF** → **`ollama create`** → **`LORA_MODEL`**.

Weights are gitignored (`*.safetensors`, `*.gguf`). Run these steps locally with GPU/disk space.

---

## 1. Merge LoRA into the base model

Produces a full HF snapshot under `models/devmentor_merged/` (configured in `.gitignore`).

**Recommended on a 12 GB GPU:**

```powershell
python finetune/merge_lora.py `
  --adapter-path finetune/adapter_test `
  --output-dir models/devmentor_merged `
  --use-4bit-base
```

If you trained into a checkpoint only:

```powershell
python finetune/merge_lora.py `
  --adapter-path finetune/adapter_test/checkpoint-125 `
  --output-dir models/devmentor_merged `
  --use-4bit-base
```

**If you have enough VRAM (~14 GB BF16)** you can omit `--use-4bit-base` and optionally pass `--dtype bf16`.

Verify: `models/devmentor_merged/config.json` exists.

---

## 2. Convert merged HF → GGUF (llama.cpp)

Clone [llama.cpp](https://github.com/ggerganov/llama.cpp) and build `quantize` (and optional tools) following its README.

Windows (PowerShell), after merge:

```powershell
$env:LLAMA_CPP = "C:\path\to\llama.cpp"
python finetune/ollama/export_gguf.py `
  --merged-dir models/devmentor_merged `
  --out-file models/devmentor-f16.gguf `
  --outtype f16
```

That runs `convert_hf_to_gguf.py` from your clone. Your layout may vary; adjust `LLAMA_CPP` until `convert_hf_to_gguf.py` is found under it.

### Quantize (recommended)

Smaller GGUF loads faster on Ollama:

```powershell
# Example: paths depend on where you built llama.cpp (Release vs CMake)
.\llama.cpp\build\bin\Release\llama-quantize.exe .\models\devmentor-f16.gguf .\models\devmentor-q4_k_m.gguf q4_K_M
```

Unix:

```bash
./llama-quantize devmentor-f16.gguf devmentor-q4_k_m.gguf q4_K_M
```

Copy `models/devmentor-q4_k_m.gguf` into `finetune/ollama/` next to the Modelfile (or edit paths below).

---

## 3. Register the model in Ollama

Place `finetune/ollama/Modelfile.sample` beside your quantized GGUF, rename/copy:

- `Modelfile.sample` → `Modelfile`
- Ensure `FROM ./your-file.gguf` matches your GGUF filename

From **`finetune/ollama/`**:

```powershell
ollama create devmentor-lora -f Modelfile
```

List models:

```powershell
ollama list
```

Quick test:

```powershell
ollama run devmentor-lora "Explain a Python IndexError in one paragraph."
```

---

## 4. Point DevMentor Mode C at the Ollama model

In your project `.env` (not committed):

```env
OLLAMA_MODEL=qwen2.5-coder:7b
LORA_MODEL=devmentor-lora
```

Restart Ollama if needed and run UI **Mode C — Base + RAG + LoRA**.

---

## Troubleshooting

| Issue | Hint |
|--------|------|
| Merge OOM | Use `--use-4bit-base` or merge on Colab/Drive then copy merged folder |
| GGUF conversion error | Ensure llama.cpp is recent enough for **Qwen2** arch; update llama.cpp |
| `ollama create` fails | Paths in `FROM` must be relative to Modelfile; use `./file.gguf` |
| Slow first run | Ollama will load GGUF once; VRAM dependent on GGUF quantization |
