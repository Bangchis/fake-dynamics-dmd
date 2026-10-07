# Fake Dynamics DMD — SDXL 4-step research implementation

Codebase triển khai bản bàn giao nghiên cứu về **teacher-anchored fake dynamics**:
fake F vừa fit samples online từ G, vừa nhận teacher-output anchor nhỏ; raw F tạo
trajectory cho consistency, còn corrected F phục vụ distribution matching.

**Trạng thái bàn giao:** đã viết code và rà soát tĩnh. Theo yêu cầu chủ repo, **chưa
chạy unit tests, toy smoke, SDXL smoke, training, checkpoint thật hoặc benchmark**
trên Mac của tác giả. Không có kết quả chứng minh FID < 17.80. Bên nhận cần chạy các
bước kiểm tra bên dưới trên máy phù hợp trước run dài.

Đọc [bản bàn giao cho bên nhận](docs/HANDOFF.md) trước, sau đó
[đặc tả gốc tiếng Việt](docs/original_handoff_vi.md),
[method và bảng nối code](docs/METHOD.md),
[runbook](docs/RUNBOOK.md), [debug](docs/DEBUGGING.md),
[evaluation](docs/EVALUATION.md) và [verification status](docs/VERIFICATION.md).
Đặc tả gốc là tài liệu nguồn; những chỉ dẫn chạy GPU trong tài liệu không có nghĩa
là tác giả đã thực hiện các run đó.

**Bên nhận có 2 × H100 80GB:** dùng [config + kế hoạch chạy đã chốt](docs/H100_RUN_PLAN.md)
và [nguồn paper / lý do chọn thông số](docs/CONFIG_SOURCES.md). Có launcher
smoke/resume → sweep CD 3 × 300 G tùy chọn → pilot 1500 G → main 5000 tổng G,
TensorBoard, log từng rank và retention một latest checkpoint mỗi run. Metric
libraries/caches có sẵn được tái dùng qua adapter; xem EVALUATION.md.

## Những phần đã triển khai

- Prompt-only SDXL conditioning đủ hai text encoders, pooled embeddings và time IDs.
- Load strict cặp checkpoint DMD2 full 019000; chỉ extract `fake_unet.*` cho F.
- CA/DM với noise và timestep riêng, corrected fake, normalization per sample và
  differentiable proxy loss vào G.
- Mixed epsilon target cho F; raw-F DDIM hai substeps; EMA-G FP32, bỏ CD tại 249.
- 5 successful F updates → 1 successful G update; counter/ramp mới bắt đầu từ 0.
- Native AMP skip handling, log JSONL, fixed probes, checkpoint atomic và resume
  model/optimizer/scaler/RNG/prompt order/cycle progress.
- TensorBoard loss/gradient ratio/drift/VRAM/timing/fixed G+EMA images; import điểm
  đánh giá thật. Opt-in optimizer stage-1 sharing và CPU FP32 EMA cho 2 H100.
- Sampler G stochastic re-noising 4 anchors; PNG, mapping và evaluation manifest.
- FID/CLIP bridge tới evaluator DMD2 ghim commit. ImageReward/HPS nhận qua metric
  plugins có provenance; chưa xác minh hoặc tự động cài ba evaluator này.

Single CUDA hoặc DDP được viết. H100 config chia AdamW state, giữ EMA master ở CPU
và teacher BF16; G/F vẫn full-weight trên mỗi rank. **Chưa đo VRAM/throughput**.
Không có FSDP, LoRA hay gradient accumulation. Chọn flag chưa hỗ trợ sẽ báo lỗi.
VAE chỉ load khi decode ảnh, không nằm trong latent training loop. Không có LMDB,
real-image train data, discriminator hay GAN objective.

## Bắt đầu trên máy của bên nhận

Python 3.11; CUDA/PyTorch tương thích GPU thực tế. Ví dụ cài CUDA 12.4 wheel:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[sdxl,dev,logging]'
fdmd plan --config configs/sdxl.yaml
fdmd doctor --config configs/sdxl.yaml
```

`plan` không load model. `doctor` chỉ đọc môi trường; không train hoặc smoke.
Top-level dependencies được ghim; ghi lại `pip freeze` thực tế khi bàn giao kết quả.
Không cần thiết lập những bước này trên Mac tác giả.

Chạy checks trên **máy bên nhận** (hoặc workflow GitHub manual):

```bash
ruff check .
ruff format --check .
pytest -q
fdmd --debug train --config configs/toy.yaml
```

Toy chỉ kiểm tra implementation. Chuẩn bị paired weights, prompts và runtime config
theo [RUNBOOK](docs/RUNBOOK.md) trước `scripts/smoke_gpu.sh` và run SDXL thật.
Không dùng defaults để suy đoán tài nguyên hoặc batch 128 từ paper.

## Nguồn và phạm vi benchmark

Sampler/checkpoint semantics đối chiếu [DMD2 tại commit
8d8fa55](https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f).
Đây là package mới với adapter SDXL minh bạch, không phải sửa trực tiếp trainer
DMD2 cũ hoặc verified reproduction toàn bộ Decoupled DMD.
Mục tiêu FID < 17.80 và bảng paper được giữ trong đặc tả người dùng; protocol
equivalence vẫn chưa xác minh. Continued training bỏ adversarial objective; warm
start DMD2 có lịch sử GAN. Không mô tả là “GAN-free from scratch”.

Code phát hành theo CC BY-NC-SA 4.0, xem [LICENSE](LICENSE) và
[attribution](NOTICE.md). Pretrained weights/data có điều khoản riêng, không nằm
trong Git repo.
