# LLM Inference Experiments: Benchmarking, Quantization, KV Cache, and Speculative Decoding

## 1. Aim

This report consolidates the inference experiments performed on an NVIDIA RTX 5060 with 8 GB VRAM under WSL2. The experiments were designed to answer four practical questions:

1. Which inference framework provides the best speed and reliability on an 8 GB consumer GPU?
2. How much memory and generation speed can be gained through model quantization?
3. How does KV-cache precision affect long-context memory usage and output quality?
4. Can importance-matrix quantization and speculative decoding improve performance without unacceptable quality loss?

The experiments covered:

- Hugging Face Transformers with native PyTorch and `bitsandbytes`
- vLLM and SGLang engine behavior
- `llama.cpp` GGUF conversion and quantization
- Standard K-quants and importance-matrix (`i-matrix`) quants
- FP16, Q4, IQ3, and IQ2 weight formats
- `f16`, `q8_0`, and `q4_0` KV-cache formats
- Speculative decoding with a smaller draft model

> **Security note:** Credentials and access tokens are intentionally excluded from this report. Authentication values should be stored in environment variables or a secret manager, never in Markdown, source code, or shell history.

---

## 2. Experimental Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 |
| VRAM | 8 GB |
| GPU architecture | Blackwell, `sm_120` |
| Operating system | Ubuntu 22.04/24.04 through WSL2 |
| CUDA | CUDA 12.8+ in the documented runs |
| Python | 3.10+; several benchmark environments used Python 3.11/3.12 |
| Frameworks | PyTorch, Transformers, bitsandbytes, vLLM, SGLang, `llama.cpp` |
| Primary models | Llama 3.1 8B Instruct; Qwen 2.5 1.5B Instruct |
| Draft model | Qwen 2.5 0.5B Instruct |
| Main context tests | 2,048 tokens for vLLM Qwen tests; 16,384 tokens for KV-cache tests |

The measurements are local engineering benchmarks rather than a standardized public leaderboard. Results depend on driver versions, compiler versions, model revisions, prompts, warm-up behavior, and whether engine initialization is included in timing.

---

## 3. Executive Summary of Results

### 3.1 Main conclusions

- **Best small-model serving result:** vLLM generated Qwen 2.5 1.5B at **102.71 tokens/sec**, compared with **60.16 tokens/sec** for the native Hugging Face baseline.
- **Best low-memory `llama.cpp` model format:** `IQ3_M` with an importance matrix used approximately **735 MB** and generated approximately **182.3 tokens/sec**, while retaining coherent output.
- **Best standard quantization balance:** `Q4_K_M` reduced the Qwen model from approximately **2.92 GB** to approximately **0.91–1.10 GB** and increased generation speed from approximately **22.7** to **59–59.9 tokens/sec**.
- **Practical KV-cache choice:** `q8_0` reduced cache memory substantially while preserving coherent output. `q4_0` produced severe degeneration in the 16k-context experiment.
- **Extreme compression limit:** `IQ2_XXS` was slightly faster and smaller than `IQ3_M`, but suffered repetitive loops, hallucinated identities, and loss of reliable reasoning.
- **8 GB Llama 3.1 deployment:** Hugging Face NF4 succeeded at approximately **30.20 tokens/sec** and **5.73 GB peak VRAM**. vLLM AWQ repeatedly failed to reserve a usable KV-cache block pool for the 8B model.
- **Speculative decoding:** Combining an `IQ3_M` target, `q8_0` KV cache, and a Qwen 0.5B draft model produced the strongest combined pipeline result documented here: approximately **189.4 tokens/sec** generation and **935.8 tokens/sec** prompt processing.

### 3.2 Recommended configurations

| Use case | Recommended configuration | Reason |
|---|---|---|
| Qwen 1.5B, highest framework throughput | vLLM, `gpu_memory_utilization=0.80`, `max_model_len=2048` | Highest measured framework throughput with a stable small model |
| Qwen 1.5B, low memory and local CLI use | `llama.cpp` with `IQ3_M` | Approximately 735 MB with coherent output |
| Qwen 1.5B, simple and reliable quantization | `llama.cpp` with `Q4_K_M` | Strong quality, simple workflow, approximately 60 tokens/sec |
| Long-context Qwen execution | `IQ3_M` weights with `q8_0` key/value cache | Approximately 50% cache reduction without observed quality loss |
| Llama 3.1 8B on 8 GB VRAM | Hugging Face with NF4 `bitsandbytes` | Fits reliably where vLLM static cache allocation did not |
| Maximum tested `llama.cpp` throughput | `IQ3_M` + `q8_0` KV cache + Qwen 0.5B speculative draft | Highest combined pipeline result |

---

## 4. Comparative Results

### 4.1 Framework comparison: Qwen 2.5 1.5B

Prompt: “Explain the mechanics of paged attention in high performance systems.”  
Output target: 64 generated tokens.

| Framework | Generation time | Throughput | Peak VRAM | Result |
|---|---:|---:|---:|---|
| Hugging Face native PyTorch | 1.0638 s | **60.16 tokens/sec** | **2.92 GB** | Stable baseline |
| vLLM v1 engine | 0.6231 s | **102.71 tokens/sec** | ~3.20 GB | **1.71x faster than HF** |
| SGLang | Not completed | Not measured | Not measured | Blocked by toolchain/JIT compatibility issues |

The vLLM result is approximately 70.7% higher throughput than the Hugging Face result for this small model and test configuration.

### 4.2 Weight quantization comparison: Qwen 2.5 1.5B

| Model format | Approx. precision | Size or VRAM | Generation speed | Quality observation |
|---|---:|---:|---:|---|
| FP16 | 16 BPW | ~2,944 MiB / 2.92 GB | ~22.7 tokens/sec | Full-precision baseline |
| `Q4_K_M` | ~4.5 BPW | ~0.91–1.10 GB | ~59.0–59.9 tokens/sec | Excellent; near-lossless in observed samples |
| `IQ3_M` with `i-matrix` | 3.66 BPW | ~735 MB | ~182.3 tokens/sec | Best quality/performance balance |
| `IQ2_XXS` with `i-matrix` | 2.06 BPW | ~481 MB | ~194.7 tokens/sec | Quantization collapse; not suitable for reliable use |

The `IQ3_M` result represents approximately 75% less model storage than FP16 while retaining coherent output. `IQ2_XXS` saves only another approximately 254 MB but crosses the practical quality limit for this 1.5B model.

### 4.3 KV-cache precision comparison at 16k context

The tests used the `IQ3_M` model with a context size of 16,384 tokens.

| KV-cache format | Generation speed | Output quality | Assessment |
|---|---:|---|---|
| `f16` | ~197.1 tokens/sec | Clean and coherent | Quality baseline; highest cache memory |
| `q8_0` | ~188.5 tokens/sec | Clean and coherent | Recommended; approximately 50% cache reduction |
| `q4_0` | ~189.8 tokens/sec | Severe collapse and looping | Unusable for this workload |

The small speed difference between the cache formats was less important than the quality difference. `q8_0` was the effective lower-risk setting; `q4_0` introduced too much numerical error into attention state storage.

### 4.4 Combined optimization result

| Metric | FP16 baseline | Combined pipeline | Improvement |
|---|---:|---:|---:|
| Model weight footprint | ~2,944 MiB | ~735 MiB | ~75% reduction |
| KV-cache format | `f16` | `q8_0` | ~50% cache-memory reduction |
| Prompt processing | ~25.3 tokens/sec | ~935.8 tokens/sec | ~37x |
| Token generation | ~22.7 tokens/sec | ~189.4 tokens/sec | ~8.3x |
| Output coherence | Baseline | Retained in observed test | No observed degradation |

The combined result used `IQ3_M` weights, `q8_0` key/value caches, a 16k context, and a Qwen 2.5 0.5B Q4 draft model with speculative decoding.

### 4.5 Llama 3.1 8B on an 8 GB GPU

| Runtime and model | Configuration | Result |
|---|---|---|
| Hugging Face + NF4 | `bitsandbytes` 4-bit; batch size 1 | ~30.20 tokens/sec; ~5.73 GB peak VRAM |
| vLLM + unquantized weights | 16-bit Llama 3.1 8B | Impossible on 8 GB; weights alone require approximately 15–16 GB |
| vLLM + AWQ | AWQ 4-bit; utilization and context tuned | Repeated `No available memory for the cache blocks` failures |

The distinction is architectural: Hugging Face allocates KV memory dynamically for the active request, while vLLM reserves engine workspaces and a static paged KV-cache pool at startup. The latter is valuable for serving and batching, but leaves too little headroom for this 8 GB Llama configuration.

---

## 5. Concepts and Rationale

### 5.1 Benchmarking methodology

- **Throughput:** Generated tokens divided by generation time.
- **Peak VRAM:** Maximum CUDA memory allocated during a measured run.
- **Prompt processing speed:** Speed of ingesting the input prompt before generation.
- **TTFT:** Time to first token, measured separately in the planned Hugging Face benchmark.
- **Warm-up:** Initial generations were run before timing to reduce one-time compilation and kernel-selection effects.
- **CUDA events:** Used in the Hugging Face tests for GPU-side generation timing.
- **Identical prompts and token counts:** Used where possible to make framework comparisons more meaningful.

These metrics answer different questions. A framework can have high generation throughput but poor startup time, high prompt-processing throughput but poor long-context behavior, or low model memory but high engine overhead.

### 5.2 Model quantization

Quantization stores weights with fewer bits. This reduces model size, memory traffic, and often the time required to read weights during autoregressive generation.

- **FP16:** High-quality baseline with large memory use.
- **Q4_K_M:** A standard mixed-precision K-quant that provides a strong quality and size trade-off.
- **IQ3_M:** An importance-aware approximately 3-bit format that protects more sensitive weights.
- **IQ2_XXS:** An aggressive approximately 2-bit format that exposes the quality floor of a small model.

The experiments showed that generation on the small Qwen model is strongly affected by memory bandwidth. Reading a much smaller quantized model on every token step can improve generation speed even when the arithmetic workload is similar.

### 5.3 Importance Matrix (`i-matrix`)

An importance matrix is produced by running representative calibration text through the model and measuring which weight values and channels are more sensitive to quantization error. The quantizer can then preserve important regions more carefully.

The calibration process used `wikitext-2`-style text and reported a baseline perplexity of **10.5976**. GPU offloading changed the calibration runtime from an estimated three hours on CPU to approximately 35 minutes.

The purpose was not merely to make the file smaller. It was to spend the remaining precision where it has the greatest effect on model behavior, allowing `IQ3_M` to remain coherent at a lower bit rate than a uniform quantization strategy would typically permit.

### 5.4 KV cache

During generation, attention stores keys and values from previous tokens. This KV cache grows with context length and can become a larger memory consumer than the quantized model weights.

- `f16` preserves the most numerical precision.
- `q8_0` approximately halves cache storage with little observed quality impact.
- `q4_0` saves more memory but introduced numerical noise severe enough to break attention tracking in the 16k experiment.

Weight quantization and KV-cache quantization are separate decisions. A small model does not automatically make an aggressively quantized KV cache safe.

### 5.5 PagedAttention and serving engines

vLLM uses PagedAttention and preallocated cache blocks to support high-throughput serving, continuous batching, and efficient sequence management. This explains the strong Qwen throughput result, but it also creates startup memory requirements.

Hugging Face generation is more request-oriented and can allocate only the memory required by the active sequence. This is why it succeeded for the 8B Llama test even though vLLM could not reserve a cache pool on the same physical GPU.

### 5.6 CUDA graphs, compilation, and engine overhead

vLLM can reserve memory for CUDA graphs, PyTorch compilation/Inductor workspaces, IPC, and its background engine process. These allocations improve serving performance in suitable environments but reduce the memory available for model weights and KV blocks.

### 5.7 Speculative decoding

Speculative decoding uses a smaller draft model to propose several tokens. The larger target model verifies those tokens in batches, potentially reducing the number of expensive target-model generation steps.

The tested pairing used:

- Target: Qwen 2.5 1.5B `IQ3_M`
- Draft: Qwen 2.5 0.5B `Q4_K_M`
- Maximum draft length: 5 tokens
- GPU offloading: `-ngl 99 -ngld 99`

The draft model must be sufficiently compatible with the target model. A faster draft is not useful if its proposals are rejected too often.

### 5.8 Blackwell compatibility and backward compilation

The RTX 5060 exposes a newer architecture than some installed CUDA and precompiled package components fully support. The experiments therefore tested compatibility fallbacks, including compiling `llama.cpp` for a supported architecture target (`89`) when direct `compute_120a` compilation was rejected.

This is a compatibility workaround, not native `sm_120` code generation. It should be re-evaluated after upgrading the CUDA toolkit, compiler, and dependent libraries.

---

## 6. Errors, Roadblocks, and Resolutions

### 6.1 Model download and cache failures

**Error:** `IncompleteSnapshotError` and inability to connect to `huggingface.co` while using `local_files_only=True`.

**Cause:** An interrupted download left partial model shards in the local cache.

**Resolution:** Resumed the snapshot download, completed the missing shards, and only then reran local-only loading.

### 6.2 PyTorch audio/CUDA package mismatch

**Error:** A TorchAudio CUDA version mismatch, including CUDA 13.0 versus CUDA 12.8 builds.

**Cause:** Packages in the virtual environment were built against different CUDA runtime versions.

**Resolution:** Reinstalled matching PyTorch-family wheels from the appropriate PyTorch wheel index.

### 6.3 `torchao` and FlashInfer architecture warnings

**Error:** Failure to load `torchao/_C_mxfp8.so` and warnings that the available binaries did not support `sm_120`.

**Cause:** Precompiled binaries targeted older GPU architectures, such as Hopper, rather than Blackwell.

**Resolution:** Allowed the runtime to fall back to standard PyTorch kernels and compatible attention implementations. The warnings did not prevent the measured baseline runs.

### 6.4 vLLM process freeze and ZeroMQ handshake timeout

**Error:** vLLM appeared to hang after `torch.compile` and eventually raised a `KeyboardInterrupt` in the ZeroMQ socket layer.

**Cause:** The vLLM v1 engine uses a background engine process and IPC handshake. Heavy compilation under WSL2 delayed readiness.

**Resolution:** Increased `VLLM_ENGINE_READY_TIMEOUT_S` to 120 seconds. This allowed startup to proceed to the subsequent memory-allocation stage.

### 6.5 vLLM cache-block allocation failure

**Error:** `ValueError: No available memory for the cache blocks`.

**Causes observed:**

- Unquantized Llama 3.1 8B weights exceeded the physical VRAM budget.
- AWQ weights still left too little space after WSL2, driver, runtime, and engine overhead.
- The model's default 128k context caused an impractically large cache requirement.
- Reducing the context to 2,048 tokens did not remove all CUDA graph, profiling, and engine allocations.
- Disabling CUDA graphs with `enforce_eager=True` still left insufficient space for the v1 engine's static cache blocks.

**Resolution and conclusion:** Lowering `gpu_memory_utilization`, reducing `max_model_len`, and disabling CUDA graphs were tested, but none made Llama 3.1 8B AWQ reliable on the 8 GB card. The practical resolution was to use Hugging Face NF4 for single-batch local inference, or move to a smaller model/`llama.cpp` for constrained deployment.

### 6.6 SGLang DeepGEMM/JIT incompatibility

**Error:** SGLang required toolchain overrides and encountered JIT compiler assertions related to `NVCC >= 12.9` and DeepGEMM on `sm_120`.

**Cause:** SGLang's optimized kernels target newer or data-center-oriented environments more directly than the local consumer GPU toolchain supported.

**Resolution attempted:** Set:

```python
import os
os.environ["SGLANG_DISABLE_DEEP_GEMM"] = "1"
os.environ["SGLANG_ALLOW_OVERRIDE_SM"] = "1"
```

The compatibility path was high friction and did not produce a comparable completed benchmark. vLLM was the more practical engine for the measured Qwen workload.

### 6.7 Unsupported CUDA architecture during `llama.cpp` compilation

**Error:** `nvcc fatal: Unsupported gpu architecture 'compute_120a'`.

**Cause:** The installed CUDA compiler did not recognize the requested Blackwell architecture target.

**Resolution:** Reconfigured the build with CUDA enabled and a supported compatibility target:

```text
-DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=89
```

This produced usable CUDA kernels for the experiments.

### 6.8 Calibration was too slow on CPU

**Error:** The `llama-imatrix` calibration run estimated approximately three hours for 4,918 chunks.

**Cause:** Model computation was not being fully offloaded to the GPU.

**Resolution:** Rebuilt `llama.cpp` with CUDA support and used `-ngl 99`. Calibration completed in approximately 35 minutes.

### 6.9 Calibration dataset download authentication and URL changes

**Error:** Direct dataset downloads returned `401 Unauthorized` or `404 Not Found`.

**Cause:** Repository access requirements and upstream URL layouts had changed.

**Resolution:** Retrieved the dataset archive from a verified alternative source, extracted it locally, and used the local file for calibration. Authentication material was not embedded in the final report.

### 6.10 KV-cache quality collapse at 4-bit precision

**Error:** `q4_0` KV-cache runs produced random character streams, prompt looping, and loss of structural coherence.

**Cause:** Four-bit cache quantization introduced too much numerical error into the key/value states used by self-attention.

**Resolution:** Selected `q8_0` as the practical lower bound for the tested 1.5B model and 16k context. It retained coherent output while reducing cache memory substantially.

### 6.11 Speculative decoding CLI flag deprecation

**Error:** The legacy `--draft` option failed with the current `llama.cpp` build.

**Cause:** The command-line interface had changed.

**Resolution:** Replaced the legacy option with the current `--spec-draft-n-max 5` syntax and used an explicit draft model path.

---

## 7. Reproducibility Notes

The following controls are important when repeating the experiments:

1. Record the exact model revision, quantized file, CUDA version, driver version, and `llama.cpp` commit.
2. Use the same prompt, prompt length, output token count, sampling parameters, and context length for each comparison.
3. Warm up kernels before collecting timed measurements.
4. Separate model-load/startup time from prompt processing and token generation time.
5. Reset CUDA peak-memory statistics before each measured run.
6. Report whether VRAM means allocated memory, reserved memory, or total device usage.
7. Repeat measurements and report variance where the result will be used for capacity planning.
8. Keep credentials outside the repository and pass them through environment variables or a secure credential store.

---

## 8. Final Conclusions

The experiments demonstrate that there is no single best inference stack for every workload on an 8 GB GPU.

- For a small model and throughput-oriented serving, **vLLM** provided the best measured framework result.
- For single-request Llama 3.1 8B inference, **Hugging Face NF4** was more reliable than vLLM because it avoided the same static cache reservation requirements.
- For the lowest memory footprint with strong quality, **`llama.cpp` `IQ3_M` with an importance matrix** was the most effective weight format.
- For long context, **`q8_0` KV caching** was a successful memory optimization, while `q4_0` crossed the quality boundary.
- For maximum tested `llama.cpp` performance, **importance-matrix weights, `q8_0` KV cache, and speculative decoding** worked well together.
- The central engineering trade-off is between memory footprint, quality retention, startup overhead, static cache capacity, and throughput. The correct configuration depends on whether the workload is a single local request, a long-context session, or a concurrent serving system.
