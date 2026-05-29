# BFP Rotation Experiment Report

이 문서는 `bfp_refactor`에서 수행한 BFP 양자화 기반 rotation 학습/평가 실험을 정리한다. 실험의 목적은 LLaMA 계열과 OPT 계열 모델에 대해 weight, activation, KV cache, 일부 attention matmul을 BFP 방식으로 fake quantization하고, rotation 학습이 perplexity에 미치는 영향을 확인하는 것이다.

코드는 기존 SpinQuant 구현에서 필요한 부분을 `bfp_refactor/` 아래로 복사해 독립적인 실험 경로로 구성했다. repo root의 legacy `train_utils`, `utils`, Executorch, GPTQ, INT quantization 경로는 이 실험의 주 대상이 아니다.

## 1. Method

### 1.1 Block Floating Point Quantization

본 실험에서 사용하는 BFP는 block 단위로 shared exponent scale을 잡고, 각 원소의 mantissa를 정해진 bit 수로 fake quantize한다. 기본 block size는 32이다.

```text
scale = 2 ^ floor(log2(absmax(block)))
q = round(abs(x) / scale * denom)
q = clamp(q, 0, 2^bits - 1)
y = sign(x) * scale * q / denom
```

여기서 `denom = (2^bits) / 2`이다. 예를 들어 4-bit BFP에서는 mantissa quant 값의 범위가 `0..15`이고, sign은 별도로 적용된다. 즉 hidden bit을 포함한 mantissa의 상위 bit를 round-to-nearest 방식으로 근사한다.

이 구현은 실제 low-bit kernel이 아니라 fake quantization이다. 따라서 메모리 절감 목적의 압축 저장이 아니라, BFP 수치 효과를 학습/평가 그래프 안에서 시뮬레이션하는 방식이다.

### 1.2 Quantization Targets

학습과 평가에서 BFP가 적용되는 대상은 다음과 같다.

| Target | LLaMA | OPT | 설명 |
| --- | --- | --- | --- |
| Weight | 지원 | 지원 | Rotation이 적용된 linear weight를 BFP fake quantization |
| Activation | 지원 | 지원 | Linear input activation을 BFP fake quantization |
| K cache | 지원 | 지원 | K tensor를 rotation 이후 BFP fake quantization |
| V cache | 지원 | 지원 | `v_proj` output을 BFP fake quantization |
| QK matmul operand | 지원 | 미지원 | LLaMA attention QK matmul 직전 operand BFP |
| AV matmul operand | 지원 | 미지원 | LLaMA attention AV matmul 직전 operand BFP |

QK/AV matmul BFP의 기본 bit는 `kv_bits`를 따른다. 메모리 peak를 줄이거나 해당 실험을 끄려면 `QK_MATMUL_BITS=16 AV_MATMUL_BITS=16`을 사용한다.

### 1.3 LLaMA Rotation

LLaMA 경로는 기존 SpinQuant의 rotation 구조를 유지한다.

| Rotation | 역할 |
| --- | --- |
| `R1` | hidden dimension 전체 activation/weight rotation |
| `R2` | attention head dimension 단위 V/O rotation |
| `R3` | RoPE 이후 Q/K online Hadamard rotation |
| `R4` | `down_proj` 입력 online Hadamard rotation |

`R3`와 `R4`는 서로 독립적으로 full/head Hadamard 또는 block-diagonal Hadamard를 선택할 수 있다.

```text
-1: full/head Hadamard
32: 32x32 block-diagonal Hadamard
```

저장 파일명 suffix는 `W_down`, `QK` 순서로 붙는다.

```text
F = full/head Hadamard
B = block-diagonal Hadamard
```

예시:

```text
R_4_4_4_FF.bin  # W_down full, QK full
R_4_4_4_BB.bin  # W_down block, QK block
R_4_4_4_FB.bin  # W_down full, QK block
```

### 1.4 OPT Rotation

OPT 경로도 `R1`과 layer별 `R2`를 사용하도록 확장했다.

| Module | Rotation |
| --- | --- |
| `q_proj`, `k_proj`, `v_proj`, `fc1` | `W @ R1` |
| `out_proj`, `fc2` | `R1.T @ W` |
| `v_proj` | head dimension block마다 `R2` 적용 |
| `out_proj` | input head dimension block마다 `R2`의 상쇄 rotation 적용 |
| 첫 decoder layer 입력 | hidden state에 `R1` 적용 |
| `lm_head` 입력 | hidden state에 `R1.T` 적용 |

OPT는 현재 `train_rotation.sh` non-FSDP 경로만 지원한다. `train_rotation_fsdp.sh`로 OPT를 실행하면 명시적으로 에러를 낸다.

### 1.5 Rotation Checkpoint

Rotation 파일은 다음 규칙으로 저장된다.

```text
bfp_runs/<model-name>/R_<w_bits>_<a_bits>_<kv_bits>_<hadamard_suffix>.bin
```

예시:

```text
bfp_runs/Llama-2-7b-hf/R_4_4_4_FF.bin
bfp_runs/opt-1.3b/R_4_4_4_FF.bin
```

저장 내용은 모델 계열에 따라 다르다.

| Model | Stored keys |
| --- | --- |
| LLaMA | `R1`, `model.layers.{i}.self_attn.R2` |
| OPT | `R1`, `model.decoder.layers.{i}.self_attn.R2` |

## 2. Experimental Setup

### 2.1 Models

실험 대상으로 고려한 모델은 다음과 같다.

| Family | Model examples | Training script |
| --- | --- | --- |
| LLaMA 2 | `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf` | 7B: `train_rotation.sh`, 13B+: `train_rotation_fsdp.sh` |
| LLaMA 3 | `meta-llama/Meta-Llama-3-8B` | `train_rotation.sh` |
| LLaMA 3.2 | `meta-llama/Llama-3.2-1B` | `train_rotation.sh` |
| LLaMA 1 | `huggyllama/llama-13b`, `huggyllama/llama-30b` | 13B/30B: `train_rotation_fsdp.sh` |
| OPT | `facebook/opt-1.3b` | `train_rotation.sh` only |

### 2.2 Datasets and Metric

평가 metric은 perplexity이다. 기본 평가는 Wikitext2이고, C4도 선택적으로 평가한다.

```bash
DATASET=wikitext2
DATASET=c4
```

평가 sample 수는 기본적으로 `EVAL_NSAMPLES=256`이다.

### 2.3 Default Training Hyperparameters

기본 학습 설정은 다음과 같다.

| Parameter | Default |
| --- | --- |
| `MAX_STEPS` | 100 |
| `MAX_LENGTH` | 2048 |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 1 |
| `LR` | 1.5 |
| `BFP_GROUP_SIZE` | 32 |
| `MODEL_DTYPE` | `auto` |
| `ROTATION_COMPUTE_DTYPE` | `fp64` |

`MODEL_DTYPE=auto`는 model config의 `torch_dtype`를 따른다. LLaMA 3 계열은 보통 BF16, LLaMA 1/2 계열은 보통 FP16으로 동작한다. config dtype이 없는 경우 FP16으로 fallback한다.

`ROTATION_COMPUTE_DTYPE=fp64`는 기존 SpinQuant 재현을 우선하는 기본값이다. 큰 모델에서 OOM이 발생하면 `ROTATION_COMPUTE_DTYPE=fp32`를 사용해 rotation matmul 임시 tensor 크기를 줄인다.

### 2.4 Bit Configurations

스크립트의 기본 인자 순서는 다음과 같다.

```text
model w_bits a_bits kv_bits
```

대표 configuration은 다음과 같다.

| Config | 의미 |
| --- | --- |
| `4 4 4` | Weight, Activation, KV 모두 4-bit BFP |
| `16 4 4` | Weight는 원본 precision, Activation/KV만 4-bit BFP |
| `16 16 16` | 대부분 원본 precision에 가까운 sanity check |

### 2.5 Hadamard Settings

기본은 full/head Hadamard이다.

```bash
W_DOWN_HAD_GROUP_SIZE=-1
QK_HAD_GROUP_SIZE=-1
```

32 block-diagonal Hadamard를 실험하려면 다음처럼 설정한다.

```bash
W_DOWN_HAD_GROUP_SIZE=32 QK_HAD_GROUP_SIZE=32 bash bfp_refactor/scripts/train_rotation.sh meta-llama/Llama-2-7b-hf 4 4 4
```

이 경우 rotation 파일명은 `R_4_4_4_BB.bin`이 된다.

### 2.6 Commands

LLaMA 2 7B, W4 A4 KV4:

```bash
bash bfp_refactor/scripts/train_rotation.sh meta-llama/Llama-2-7b-hf 4 4 4
bash bfp_refactor/scripts/eval_ppl.sh meta-llama/Llama-2-7b-hf bfp_runs/Llama-2-7b-hf 4 4 4
```

LLaMA 2 13B FSDP:

```bash
NPROC_PER_NODE=2 bash bfp_refactor/scripts/train_rotation_fsdp.sh meta-llama/Llama-2-13b-hf 4 4 4
```

LLaMA 3.2 1B:

```bash
NPROC_PER_NODE=1 bash bfp_refactor/scripts/train_rotation.sh meta-llama/Llama-3.2-1B 4 4 4
```

OPT 1.3B:

```bash
NPROC_PER_NODE=1 bash bfp_refactor/scripts/train_rotation.sh facebook/opt-1.3b 4 4 4

bash bfp_refactor/scripts/eval_ppl.sh facebook/opt-1.3b bfp_runs/opt-1.3b 4 4 4
```

Rotation 없이 BFP만 평가:

```bash
NO_ROTATE=1 bash bfp_refactor/scripts/eval_ppl.sh meta-llama/Llama-2-7b-hf 4 4 4
```

C4 평가:

```bash
DATASET=c4 bash bfp_refactor/scripts/eval_ppl.sh   meta-llama/Llama-2-7b-hf   bfp_runs/Llama-2-7b-hf   4 4 4
```

30B memory-saving setup:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=2 MODEL_DTYPE=fp16 ROTATION_COMPUTE_DTYPE=fp32 MAX_LENGTH=1024 PER_DEVICE_TRAIN_BATCH_SIZE=1 QK_MATMUL_BITS=16 AV_MATMUL_BITS=16 bash bfp_refactor/scripts/train_rotation_fsdp.sh huggyllama/llama-30b 4 4 4
```

## 3. Experimental Results

아래 표는 결과를 정리하기 위한 형식이다. 실제 PPL 값은 각 run의 출력값을 채워 넣는다.

### 3.1 Wikitext2 PPL

| Model | Method | W/A/KV | Hadamard | QK/AV matmul BFP | Rotate | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-2-7B | FP baseline | 16/16/16 | FF | Off | Off | TBD |
| Llama-2-7B | BFP only | 4/4/4 | FF | On | Off | TBD |
| Llama-2-7B | BFP + learned rotation | 4/4/4 | FF | On | On | TBD |
| Llama-2-7B | BFP + learned rotation | 4/4/4 | BB | On | On | TBD |
| Llama-2-7B | A/KV BFP + learned rotation | 16/4/4 | FF | On | On | TBD |
| Llama-2-13B | BFP + learned rotation | 4/4/4 | FF | On | On | TBD |
| Llama-3.2-1B | BFP + learned rotation | 4/4/4 | FF | On | On | TBD |
| OPT-1.3B | BFP + learned rotation | 4/4/4 | FF | Off | On | TBD |

### 3.2 C4 PPL

| Model | Method | W/A/KV | Hadamard | QK/AV matmul BFP | Rotate | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-2-7B | FP baseline | 16/16/16 | FF | Off | Off | TBD |
| Llama-2-7B | BFP + learned rotation | 4/4/4 | FF | On | On | TBD |
| Llama-2-7B | BFP + learned rotation | 4/4/4 | BB | On | On | TBD |
| OPT-1.3B | BFP + learned rotation | 4/4/4 | FF | Off | On | TBD |

### 3.3 Ablation Axes

실험 결과는 다음 축으로 비교하는 것이 좋다.

| Ablation | 비교 |
| --- | --- |
| Rotation 효과 | `NO_ROTATE=1` vs learned rotation |
| Weight BFP 영향 | `4/4/4` vs `16/4/4` |
| Online Hadamard 크기 | `FF` vs `BB` vs `FB` |
| Matmul BFP 영향 | `QK_MATMUL_BITS=4 AV_MATMUL_BITS=4` vs `16/16` |
| dtype 영향 | `MODEL_DTYPE=auto/fp16/bf16` |
| rotation compute dtype 영향 | `ROTATION_COMPUTE_DTYPE=fp64` vs `fp32` |

## 4. Analysis and Discussion

### 4.1 Rotation의 역할

BFP는 block 단위 absmax를 공유하므로, block 내부 값의 dynamic range가 크면 작은 값의 mantissa precision이 빠르게 손실된다. Rotation은 activation과 weight의 분포를 더 균등하게 만들어 block-wise BFP에서 발생하는 outlier 영향을 줄이는 역할을 한다.

LLaMA에서는 `R1`, `R2`, `R3`, `R4`가 함께 작동한다. `R1`은 hidden dimension 전체 분포를 조정하고, `R2`는 V/O projection의 head-wise 분포를 조정한다. `R3`는 RoPE 이후 Q/K에 적용되는 online Hadamard이고, `R4`는 `down_proj` 입력에 적용되는 online Hadamard이다.

OPT는 RoPE가 없으므로 LLaMA의 R3와 동일한 Q/K after-RoPE rotation은 없다. 대신 `R1`과 layer별 `R2`를 통해 linear 및 V/O projection 쪽을 LLaMA와 유사하게 맞췄다.

### 4.2 FF vs BB Hadamard

Full/head Hadamard는 더 넓은 차원에서 mixing을 수행하므로 분포 smoothing 효과가 클 수 있다. 반면 32x32 block-diagonal Hadamard는 연산 단위가 작아 실제 구현에서는 더 빠를 가능성이 있지만, full Hadamard만큼 전역 mixing을 제공하지는 않는다.

따라서 FF와 BB의 비교는 정확도와 효율 사이의 trade-off를 확인하는 실험이다.

### 4.3 QK/AV Matmul BFP

LLaMA 경로에서는 QK matmul과 AV matmul operand에도 BFP fake quantization을 적용했다. QK matmul의 경우 rotation 이후 Q/K에 BFP가 들어가도록 구현되어 있다. 이 경로는 attention score와 context aggregation 연산 자체의 quantization 민감도를 확인하기 위한 것이다.

다만 현재 구현은 실제 BFP matmul kernel이 아니라 PyTorch tensor 연산으로 quant-dequant를 삽입하는 방식이다. 따라서 kernel launch와 임시 tensor 비용이 증가할 수 있고, 메모리와 속도 측면에서 실제 hardware BFP와 다르게 보일 수 있다.

### 4.4 Large Model Memory

30B 실험에서 OOM이 발생한 주요 원인은 전체 parameter 수 자체보다 forward 중 peak memory이다. FSDP는 parameter를 shard하지만, layer forward 시점에는 필요한 weight를 all-gather한다. 여기에 rotation을 위해 `W @ R1` 형태의 rotated weight 임시 tensor가 추가로 생긴다.

특히 기존 재현을 위해 `ROTATION_COMPUTE_DTYPE=fp64`를 사용하면 fp16 weight 대비 4배 크기의 임시 tensor가 생성된다. 30B에서는 이 peak가 140GB GPU에서도 OOM을 만들 수 있다. 이를 완화하기 위해 다음을 사용한다.

```bash
ROTATION_COMPUTE_DTYPE=fp32
QK_MATMUL_BITS=16
AV_MATMUL_BITS=16
MAX_LENGTH=1024
```

진짜로 큰 linear 연산을 GPU 2장에 나누려면 FSDP만으로는 부족하다. FSDP는 sharded storage와 all-gather 기반 data parallelism에 가깝고, `gate_proj` 같은 큰 linear 계산을 column/row 단위로 나누는 것은 tensor parallel linear 구현이 필요하다.

### 4.5 Current Limitations

- OPT FSDP는 아직 지원하지 않는다.
- OPT QK/AV matmul BFP monkeypatch는 아직 없다.
- BFP는 fake quantization이므로 실제 low-bit kernel의 memory/speed 특성을 반영하지 않는다.
- Large model에서는 rotated weight 임시 tensor가 peak memory를 지배할 수 있다.
- `ROTATION_COMPUTE_DTYPE=fp32`는 메모리 완화에 도움이 되지만, 기존 FP64 재현과 수치가 약간 달라질 수 있다.

## 5. Practical Checklist

새 실험을 추가할 때는 아래 항목을 기록한다.

```text
Model:
Dataset:
W/A/KV bits:
BFP group size:
Hadamard suffix:
QK/AV matmul bits:
MODEL_DTYPE:
ROTATION_COMPUTE_DTYPE:
MAX_LENGTH:
MAX_STEPS:
NPROC_PER_NODE:
Rotation file:
PPL:
Notes:
```

결과 table에는 최소한 baseline, no-rotation BFP, learned-rotation BFP를 함께 넣는 것이 좋다. 이렇게 해야 BFP 자체의 손실과 rotation 학습으로 회복되는 정도를 분리해서 해석할 수 있다.
