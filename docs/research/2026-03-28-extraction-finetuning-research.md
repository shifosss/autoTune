# Structured Extraction LoRA Finetuning: Research Report

**Date**: 2026-03-28
**Scope**: Dataset design, evaluation metrics, and model selection for 30-40 field structured extraction from messy unstructured text, contextualized within the autoTune framework.
**Sources**: 40+ | **Confidence**: High

---

## Executive Summary

This report defines a complete finetuning pipeline for extracting 30-40 structured fields from medium-to-long messy text using LoRA on Apple M5 Pro (64GB). The recommended stack:

- **Model**: Qwen 2.5 7B Instruct (best native JSON capability; 8-12GB QLoRA training)
- **Dataset**: 300-500 curated ChatML examples with flat JSON output; scale to 1-2K via synthetic reverse generation
- **Loss**: Standard cross-entropy on completion tokens only
- **Metrics**: JSON validity -> per-field exact/fuzzy match -> aggregate exact match -> source grounding rate
- **Training**: `mlx-lm` QLoRA, rank=16, alpha=32, lr=2e-4, 1-3 epochs

All components map directly to autoTune's overlay model, result contract, and manifold-navigated hyperparameter space.

---

## 1. Dataset Design

### 1.1 Input/Output Format

Use **ChatML (OpenAI messages format)** in JSONL files. This matches the chat template of Qwen 2.5 and all modern instruction-tuned models.

```json
{"messages": [
  {"role": "system", "content": "You are an expert at structured data extraction. You will be given unstructured text and must extract information into the provided JSON schema. Output valid JSON only. Use null for fields not found in the text."},
  {"role": "user", "content": "<RAW MESSY TEXT>\n\nExtract the following fields:\n<JSON SCHEMA TEMPLATE>"},
  {"role": "assistant", "content": "<FILLED JSON>"}
]}
```

**Critical rules:**
- `train_on_inputs: false` — loss computed only on the assistant turn (the JSON output)
- Include the empty JSON schema template in every user message (NuExtract pattern) — teaches the model to follow arbitrary schemas
- Use the model's exact chat template (BOS/EOS tokens, role names) — mismatched templates are the #1 cause of degraded post-training performance
  - Source: [Unsloth Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)

### 1.2 Output Schema Design (30-40 Fields)

Use **flat JSON with max 1-level nesting**. Group related fields (3-5 per group) under a parent key. Do not go deeper.

```json
{
  "entity_name": "...",
  "entity_type": "...",
  "contact": {
    "phone": "...",
    "email": "...",
    "address": "..."
  },
  "financial": {
    "amount": null,
    "currency": "USD",
    "date": "2026-01-15"
  },
  "category_field_1": "...",
  "category_field_2": ["item_a", "item_b"]
}
```

**Design rules:**
- **`null` for missing fields** — never omit keys. Makes evaluation unambiguous (true negative vs missed extraction)
- **Empty arrays `[]`** for absent list-valued fields (distinct from `null` = "unknown")
- **Order fields by extraction difficulty** — easier fields first; model builds momentum from early successes
- **Define schema via Pydantic/JSON Schema** — serves as both validation and prompt documentation

**Why shallow nesting**: ExtractBench ([arXiv:2602.12247](https://arxiv.org/abs/2602.12247)) found model performance degrades sharply with schema breadth and nesting depth. Frontier models hit 0% valid output on a 369-field deeply nested schema.

### 1.3 Training Data Size

| Complexity | Examples | Notes |
|-----------|---------|-------|
| Simple extraction (few fields, clean text) | 100-300 | Per Particula.tech |
| **Your use case (30-40 fields, messy text)** | **300-500 curated, scale to 1-2K** | Sweet spot for LoRA |
| Complex domain adaptation | 1,000-5,000 | Specialized vocabulary |

**Key finding**: 200 curated examples outperform 2,000 sloppy ones. A logistics company achieved 94% accuracy on invoice extraction with just 280 LoRA-finetuned examples ([Particula.tech](https://particula.tech/blog/how-much-data-fine-tune-llm)). Multi-epoch training on instruction data often hurts — stick to 1-3 epochs ([Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)).

### 1.4 Synthetic Data Pipeline (Reverse Generation)

The most powerful technique: **give the LLM the structured output and ask it to generate plausible messy input text** (SynthIE, [EMNLP 2023](https://arxiv.org/abs/2303.04132)). This exploits task asymmetry — generating text from structured data is far easier for LLMs than the reverse.

```
Step 1: Define your 30-40 field JSON schema
Step 2: Manually create 50-100 gold examples from real messy text
Step 3: Use a strong teacher model (Claude, GPT-4, Llama 405B) for reverse generation:
        "Given this JSON, write a realistic messy unstructured document that contains this information"
Step 4: Validate all outputs against JSON Schema
Step 5: Human review ~20% random sample
Step 6: Train LoRA, evaluate on held-out real examples
Step 7: Identify weak fields, generate targeted synthetic data for those
Step 8: Retrain and iterate
```

### 1.5 Existing Public Datasets (for bootstrapping or augmentation)

| Dataset | Size | Relevance | Source |
|---------|------|-----------|--------|
| **ScrapeGraphAI-100k** | 93,695 | Markdown + JSON schema + LLM response, diverse domains | [arXiv:2602.15189](https://arxiv.org/abs/2602.15189) |
| **HenriqueGodoy/extract-0** | 280,128 | Synthetic doc extraction, JSON-validated | [HuggingFace](https://huggingface.co/datasets/HenriqueGodoy/extract-0) |
| **Cleanlab/insurance-claims-extraction** | Benchmark | 15+ nested fields from insurance docs | [HuggingFace](https://huggingface.co/datasets/Cleanlab/insurance-claims-extraction) |
| **nlpaueb/finer-139** | Large | 139 entity types from financial text | [HuggingFace](https://huggingface.co/datasets/nlpaueb/finer-139) |
| **NousResearch/hermes-function-calling-v1** | Large | Structured output + function calling | [HuggingFace](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) |

**Baseline model for generating initial labels**: NuExtract 2.0 ([numind/NuExtract-2.0-8B](https://huggingface.co/numind/NuExtract-2.0-8B)) — outperforms GPT-4o on extraction while being 500x smaller.

---

## 2. Loss Function & Evaluation Metrics

### 2.1 Training Loss

**Standard cross-entropy on completion tokens only.** This is the default in all major frameworks (HuggingFace TRL `SFTTrainer`, Axolotl, mlx-lm). The prompt/input tokens are masked (label index = -100).

**Advanced option**: SFT-GO ([arXiv:2506.15021](https://arxiv.org/abs/2506.15021)) — token-importance-weighted loss that upweights value tokens (the extracted content) over JSON structural tokens (`{`, `}`, `"key":`). Marginal gain; standard CE is sufficient to start.

**At inference**: Combine LoRA adapter with **constrained decoding** (Outlines, XGrammar, llama.cpp grammars) to guarantee valid JSON. Predibase showed this combination achieves 99.9% well-formed outputs and surpasses either approach alone ([Predibase blog](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora)).

### 2.2 Evaluation Metrics

#### Metric Stack (ordered by priority)

```
Level 0 (Gate):      JSON validity rate, schema compliance rate
Level 1 (Per-field): Per-field exact match (categorical), per-field fuzzy match (text)
Level 2 (Aggregate): Macro-avg field accuracy, aggregate exact match (all fields correct)
Level 3 (Fidelity):  Source grounding rate, hallucination rate
```

#### Level 0: Structural Validity (Gate Metrics)

| Metric | What | How |
|--------|------|-----|
| **JSON Validity Rate** | % outputs parseable as JSON | `json.loads()` success rate |
| **Schema Compliance Rate** | % valid JSON conforming to target schema | `jsonschema.validate()` |
| **Required Field Presence** | % outputs with all required keys | Key presence check |

If these fail, downstream metrics are meaningless. StructEval ([arXiv:2505.20139](https://arxiv.org/html/2505.20139v1)) found even GPT-4o achieves only 76% structural correctness.

#### Level 1: Per-Field Accuracy

| Metric | When to Use |
|--------|-------------|
| **Exact Match (EM)** | Categorical fields, IDs, codes, dates |
| **Fuzzy Match** (Levenshtein / Jaccard) | Names, addresses, free-text fields |
| **Token-level F1** | Long text fields, descriptions |
| **Set-level F1** | Multi-value fields (lists/arrays), order-insensitive |

#### Level 2: Aggregate Accuracy

| Metric | What |
|--------|------|
| **Macro-avg Field Accuracy** | Average accuracy across all fields (each field weighted equally) |
| **Aggregate Exact Match** | % of samples where ALL fields are correct — the strictest metric |

Strata ([Nature 2025](https://www.nature.com/articles/s41598-025-28767-z)): LoRA-finetuned Llama 3.1 8B achieved 90.0% aggregate exact match, matching human annotator performance.

#### Level 3: Faithfulness

| Metric | What |
|--------|------|
| **Source Grounding Rate** | % of extracted values traceable to a span in the source text |
| **Hallucination Rate** | % of values with no corresponding source span |

LangExtract ([Google, 2025](https://github.com/google/langextract)) implements source grounding via character-interval mapping.

### 2.3 Handling Missing Fields

Train the model to output `null` for absent fields. Evaluate:
- **True negative**: correctly `null` (field absent from source, model outputs `null`)
- **False negative**: missed extraction (field present, model outputs `null`)
- **False positive / hallucination**: field absent, model fabricates a value

### 2.4 Handling Multi-Value Fields

Use JSON arrays. Evaluate with **set-level F1** (order-insensitive): `precision = |correct items| / |predicted items|`, `recall = |correct items| / |ground truth items|`. For fuzzy set matching, use AESOP ([Microsoft, EMNLP 2024](https://arxiv.org/html/2402.04437v3)).

---

## 3. Model Selection

### 3.1 Recommendation: Qwen 2.5 7B Instruct

| Aspect | Detail |
|--------|--------|
| **Model** | `Qwen/Qwen2.5-7B-Instruct` (MLX 4-bit: `mlx-community/Qwen2.5-7B-Instruct-4bit`) |
| **Why** | Best-in-class native JSON/structured output capability at 7B; explicitly trained on structured data understanding |
| **Parameters** | 7.6B |
| **Training memory** | ~8-12GB with QLoRA (4-bit base + FP16 adapter) |
| **Inference memory** | ~4GB (4-bit quantized) |
| **License** | Apache 2.0 |

**Why Qwen over Llama for this task**: Qwen 2.5 was explicitly optimized for structured data understanding and JSON generation. For 30-40 field extraction, this native capability gives a head start before finetuning ([Qwen 2.5 Technical Report](https://arxiv.org/pdf/2412.15115)).

**Fallback**: Llama 3.1 8B Instruct — the "most tunable" model family (biggest improvement from finetuning). Apache-compatible license. Use if Qwen's JSON performance disappoints on your specific domain.

### 3.2 Alternatives

| Model | Params | Training Memory (QLoRA) | Best For |
|-------|--------|------------------------|----------|
| **Qwen 3.5 4B** | 4B | ~4-8GB | Fastest training; matches 30x larger models after finetuning |
| **Llama 3.1 8B** | 8B | ~8-12GB | Largest improvement margin from finetuning |
| **Phi-4-mini** | 3.8B | ~4-8GB | Smallest model with strong reasoning |
| **Gemma 3 12B** | 12B | ~12-18GB | Higher capability ceiling if 7B underperforms |
| **Qwen 2.5 14B** | 14B | ~18-25GB | Maximum quality; still fits on 64GB M5 Pro |

### 3.3 Training Framework

**`mlx-lm`** — native Apple Silicon LoRA/QLoRA. Zero memory copies between CPU and GPU on unified memory architecture.

```bash
# Install
pip install mlx-lm

# QLoRA fine-tune (auto-detected from 4-bit model)
mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --train \
  --data ./training_data \
  --batch-size 4 \
  --lora-layers 16 \
  --iters 1000
```

**Confirmed compatible**: Qwen, Llama, Mistral, Phi, Gemma families all supported in mlx-lm.

### 3.4 LoRA Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Rank (r)** | 16 (start), up to 32 if underfitting | 30-40 fields is moderately complex; rank 16 is the extraction sweet spot |
| **Alpha** | 32 (2x rank) | Amplifies adapter contribution; standard for extraction tasks |
| **Target modules** | All attention layers (q_proj, k_proj, v_proj, o_proj) | Broad coverage for diverse field types |
| **Learning rate** | 2e-4 | Standard LoRA range; autoTune can navigate [1e-5, 1e-3] |
| **Dropout** | 0.05 | Light regularization; autoTune range [0.0, 0.3] |
| **Warmup ratio** | 0.06 | ~6% of training steps |
| **Batch size** | 4-8 | MLX on 64GB; batch 4 recommended for long sequences |
| **Epochs** | 1-3 | Loss stabilizes by epoch 3; more epochs often hurts |

---

## 4. Mapping to autoTune

### 4.1 config.yaml

```yaml
project:
  name: "extraction-finetuning"
  base_model: "Qwen/Qwen2.5-7B-Instruct"

task:
  type: "instruction_tuning"

run:
  script: "train.py"
  checkpoint_format: "safetensors"

retention:
  enabled: false        # enable later with a general-knowledge probe set if needed

parameters:
  lr:
    type: float
    range: [1.0e-5, 1.0e-3]
    log_scale: true
  rank:
    type: int
    range: [4, 64]
  alpha:
    type: int
    range: [8, 128]
  dropout:
    type: float
    range: [0.0, 0.3]
  warmup_ratio:
    type: float
    range: [0.0, 0.15]

scoring:
  exploration_weight: 0.005
  diversity_weight: 0.01
  retention_cliff_threshold: -0.10
```

The manifold navigator will learn correlations between these parameters (e.g., rank-alpha coupling, lr-dropout tradeoff) and propose moves along discovered principal components.

### 4.2 result.json Contract

```json
{
  "status": "completed",
  "metrics": {
    "task_loss": 0.42,
    "task_accuracy": 0.88,
    "json_validity_rate": 1.0,
    "schema_compliance_rate": 0.98,
    "aggregate_exact_match": 0.72,
    "macro_field_accuracy": 0.85,
    "source_grounding_rate": 0.96,
    "hallucination_rate": 0.02
  },
  "composite": 0.85,
  "elapsed_seconds": 1200,
  "peak_vram_mb": 10000,
  "checkpoint_saved": true
}
```

**`task_score`** should map to `aggregate_exact_match` or `macro_field_accuracy` — this is what the DAG-UCB scoring and Pareto ranking optimize. The `composite` can be a weighted combination:

```
composite = 0.6 * macro_field_accuracy + 0.3 * aggregate_exact_match + 0.1 * (1 - hallucination_rate)
```

### 4.3 Hyperparameter Space (theta)

The manifold navigates these dimensions:

| Parameter | Type | Range | Log Scale |
|-----------|------|-------|-----------|
| `lr` | float | [1e-5, 1e-3] | yes |
| `rank` | int | [4, 64] | no |
| `alpha` | int | [8, 128] | no |
| `dropout` | float | [0.0, 0.3] | no |
| `warmup_ratio` | float | [0.0, 0.15] | no |

CMA-ES will discover:
- **rank-alpha correlation** (alpha typically scales with rank)
- **lr-rank coupling** (higher ranks need lower lr to avoid collapse)
- **dropout-rank interaction** (higher ranks benefit from more regularization)

### 4.4 Retention (Optional, Phase 2)

If catastrophic forgetting of general capabilities matters, enable retention:

```yaml
retention:
  enabled: true
  script: "eval_retention.py"
  probe_set: ".autotune_retention_probe/"
```

Probe set: a small set of general instruction-following tasks (e.g., MMLU subset, general Q&A). The retention script measures how well the finetuned model preserves general capability. This activates 3-objective Pareto optimization (task x retention x cost) and forgetting cliff detection.

---

## 5. Key References

### Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [ExtractBench](https://arxiv.org/abs/2602.12247) | 2025 | 12,867-field extraction benchmark; performance degrades with nesting |
| [SynthIE](https://arxiv.org/abs/2303.04132) | 2023 | Reverse generation for synthetic extraction data (+57 F1) |
| [AESOP](https://arxiv.org/html/2402.04437v3) | 2024 | Entity-centric extraction evaluation metric |
| [Strata](https://www.nature.com/articles/s41598-025-28767-z) | 2025 | LoRA Llama-8B matches human annotators at 90% exact match |
| [SFT-GO](https://arxiv.org/abs/2506.15021) | 2025 | Token-importance-weighted loss for SFT |
| [StructEval](https://arxiv.org/html/2505.20139v1) | 2025 | Structural output benchmark; GPT-4o only 76% |
| [KIEval](https://arxiv.org/abs/2503.05488) | 2025 | Group-level document extraction evaluation |
| [JSONSchemaBench](https://arxiv.org/abs/2501.10868) | 2025 | 10K real-world JSON schemas for constrained decoding |

### Tools & Libraries

| Tool | Purpose |
|------|---------|
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | LoRA/QLoRA finetuning on Apple Silicon |
| [Outlines](https://github.com/outlines-dev/outlines) | Constrained decoding for JSON schema compliance |
| [LangExtract](https://github.com/google/langextract) | Extraction with source grounding |
| [NuExtract 2.0](https://huggingface.co/numind/NuExtract-2.0-8B) | Baseline extraction model / label generation |

### Guides

| Resource | Key Takeaway |
|----------|--------------|
| [Predibase: LoRA + Outlines](https://predibase.com/blog/lorax-outlines-better-json-extraction-with-structured-generation-and-lora) | Finetuning + constrained decoding = best combo |
| [Databricks: Structured Extraction Part 2](https://community.databricks.com/t5/technical-blog/end-to-end-structured-extraction-with-llm-part-2-fine-tuning/ba-p/99900) | Per-field fuzzy evaluation, synthetic data |
| [Raschka: LoRA Tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) | 1-3 epochs max; multi-epoch hurts |
| [Particula.tech: Data Size](https://particula.tech/blog/how-much-data-fine-tune-llm) | 280 curated examples -> 94% accuracy |

---

## Methodology

Searched 30+ queries across web, academic databases, and HuggingFace. Analyzed 40+ sources. Cross-referenced claims across multiple independent sources.

**Sub-questions investigated:**
1. Ideal input/output format for extraction finetuning data
2. Existing public datasets for multi-field extraction from messy text
3. Recommended training data size for LoRA finetuning
4. Synthetic data generation best practices
5. Output schema design for 30-40 fields
6. Loss functions for LoRA extraction finetuning
7. Per-field and aggregate evaluation metrics
8. Handling missing fields and multi-value fields
9. Small LLM candidates for M5 Pro 64GB
10. Memory requirements and quantization strategy
11. MLX compatibility and training framework
12. LoRA hyperparameter settings for extraction tasks
