# Native inference test record

- Runtime source: `ggml-org/llama.cpp`, commit `3f03b17` (0.7.0-dev).
- Build: Linux x86_64, CPU backend; `GGML_NATIVE=OFF`, SSE4.2 on, AVX/AVX2/FMA/F16C/BMI2 off; no GPU.
- Model: `model/ravan-sft-Q4_0.gguf` (621,760 bytes; 875K parameters; 128 training context).
- Test: llama.cpp loaded GGUF; server `/health` returned `{"status":"ok"}`; custom UI root returned HTTP 200; a POST to native `/completion` succeeded.
- The target user's Windows machine was not available. Build host CPU is a Xeon, not a Pentium G3250.

For `<|user|>\nسلام\n<|assistant|>\n`, the actual native response began:

```text
‌ ملهی‌ ملهی‌ ملهی‌ ملهی‌ ملهی‌ ملهی‌ ملهی‌ ملهی‌ ملهی‌
```

This is malformed output, not a pass on language quality. Native load and route are verified; useful responses are not.
