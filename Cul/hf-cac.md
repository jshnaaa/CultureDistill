# HF-CAC 运行命令

## NormAD 数据集

```bash
python Cul/generate_hf_cac_data.py \
    --input_file /autodl-fs/data/normad_mas.json \
    --output_file /autodl-fs/data/qwen/normad_hf_cac_inference.jsonl \
    --model_name qwen \
    --use_vllm --tensor_parallel_size 2 \
    --max_samples 0 --negotiation_rounds 1 \
    --include_judge true
```

## CultureAtlas 数据集

```bash
python Cul/generate_hf_cac_data.py \
    --input_file /autodl-fs/data/cultureAtlas_mas.json \
    --output_file /autodl-fs/data/qwen/cultureatlas_hf_cac_inference.jsonl \
    --model_name qwen \
    --use_vllm --tensor_parallel_size 2 \
    --max_samples 0 --negotiation_rounds 1 \
    --include_judge true
```

## CulturalBench 数据集

```bash
python Cul/generate_hf_cac_data.py \
    --input_file /autodl-fs/data/culturalBench_mas_before.json \
    --output_file /autodl-fs/data/qwen/culturalbench_hf_cac_inference.jsonl \
    --model_name qwen \
    --use_vllm --tensor_parallel_size 2 \
    --max_samples 0 --negotiation_rounds 1 \
    --include_judge true
```

## CultureLLM 数据集（WVS 态度预测）

```bash
python Cul/generate_hf_cac_data.py \
    --input_file /autodl-fs/data/cultureLLM_mas.json \
    --output_file /autodl-fs/data/qwen/culturellm_hf_cac_inference.jsonl \
    --model_name qwen \
    --use_vllm --tensor_parallel_size 2 \
    --max_samples 0 --negotiation_rounds 1 \
    --include_judge true
```

## 评估

```bash
python Cul/evaluate.py \
    --input_file /autodl-fs/data/qwen/culturellm_hf_cac_inference.jsonl \
    --max_choice 9
```

## 参数说明

- `--max_samples 0`：处理全部数据（设为正数则仅处理前 N 条）
- `--negotiation_rounds 1`：启用 MAD 式多轮辩论（0 表示仅单轮生成）
- `--include_judge true`：在分歧时启用 Judge 仲裁
- `--tensor_parallel_size 2`：张量并行数，视 GPU 数量调整
- `--config_path`：可选，手动指定配置文件路径（默认自动检测）
