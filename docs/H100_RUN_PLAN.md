# Quyết định triển khai cho 2 × H100 80GB

Đây là cấu hình và quy trình bên nhận sẽ chạy. Tác giả chỉ kiểm tra tĩnh trên Mac;
chưa có bằng chứng vừa VRAM, smoke/resume thành công hay điểm benchmark.
Nguồn cho từng thông số nằm trong [CONFIG_SOURCES.md](CONFIG_SOURCES.md).

## Cấu hình đã chốt

| Mục | Giá trị |
|---|---|
| Architecture / output | full-weight SDXL, 1024, 4 bước 999/749/499/249 |
| Initialization | cặp full DMD2 G/F 019000; optimizers mới |
| Hardware mode | 1 node, 2 ranks DDP; ZeroRedundancyOptimizer stage 1 |
| Precision / memory | G/F FP32 + BF16 autocast; teacher BF16; EMA master FP32 CPU, snapshot BF16 CUDA |
| Batch | 1/GPU → global 2; accumulation 1; activation checkpointing bật |
| LR G/F | 5e-7 / 5e-7, constant; AdamW (0.9,0.999), decay 0.01, clip 10 |
| Updates | 5 F thành công → 1 G thành công |
| Teacher CFG | 8; sampler G không dùng external CFG |
| Beta | 0 → max 0.05, bắt đầu k_G=100, đạt max ở 1100 |
| CD / EMA | CD max 0.1, ramp 100 G; EMA decay 0.99 sau G thành công |
| Pilot / main | 1500 G; tiếp tục cùng run đến **5000 tổng G**, không cộng thêm 5000 |
| Logs | JSONL từng update/rank + TensorBoard; probe mỗi 25 G; performance mỗi 10 G |
| Fixed images | 8 training prompts × G/EMA mỗi 250 G; cùng initial/re-noising seed; lưu vào TensorBoard |
| Resume checkpoints | mỗi 500 G, chỉ giữ latest; cuối run không viết trùng periodic cùng G |

DDP vẫn giữ toàn bộ G/F trên mỗi GPU; stage 1 chia AdamW state. CPU EMA chuyển
snapshot tạm sang CUDA cho targets rồi giải phóng trước backward/optimizer step.
Snapshot BF16 khác với forward FP32: phải kiểm tra sai số và CD dynamics trên máy
thật. Host RAM, CPU↔GPU transfer, NCCL và checkpoint I/O cũng cần đo. Chuẩn bị host
RAM khoảng 128GB trở lên như một mức dự trù, rồi đo RSS thực tế; đây không phải
cam kết đủ bộ nhớ. Đĩa cần chứa assets và hai checkpoint trong lúc atomic save.
Một resume checkpoint full SDXL có thể cỡ hàng chục GB; schema 2 không nhân đôi
G/F/EMA theo số rank. Sweep giữ latest của từng candidate đến khi chọn xong.

## Chuẩn bị trên máy bên nhận

Dùng env Python 3.11 với Torch 2.6.0 phù hợp CUDA. Nếu env đã có đúng dependencies,
chỉ bổ sung thành phần thiếu. Ví dụ package extras:

```bash
python -m pip install -e '.[sdxl,dev,logging]'
ruff check .
ruff format --check .
pytest -q
```

HF đã đăng nhập trên máy bên nhận thì dùng cache/login đó. Không truyền token vào
config hay repo. Không tải lại ImageReward/HPS/FID/CLIP đã có chỉ để theo ví dụ cài
mới. Shared metric cache được đọc; report ghi version/revision/hash/thang điểm.

Thiết lập ba đường dẫn thật, không dùng LoRA checkpoint:

```bash
export GENERATOR_CHECKPOINT=/absolute/path/to/pytorch_model.bin
export GUIDANCE_CHECKPOINT=/absolute/path/to/pytorch_model_1.bin
export TRAIN_PROMPTS_JSONL=/absolute/path/to/laion-prompts.jsonl
export RUN_ROOT=runs/h100
fdmd inspect-checkpoint "$GENERATOR_CHECKPOINT"
fdmd inspect-checkpoint "$GUIDANCE_CHECKPOINT"
fdmd doctor --config configs/h100_2x80.yaml
```

Giữ nguyên full checkpoints khởi tạo và asset manifest. Có thể kiểm tra kế hoạch
bằng `fdmd plan --config configs/h100_2x80.yaml`; đường dẫn còn null được launcher
điền từ env. Launcher không cài packages, fetch assets hoặc metric weights.

## Smoke, resume và chọn physical batch

```bash
bash scripts/run_h100.sh smoke
python scripts/check_smoke.py "$RUN_ROOT/smoke"
bash scripts/run_h100.sh smoke-resume
python scripts/check_smoke.py "$RUN_ROOT/smoke" --expected-updates 12
```

Smoke ép beta/CD khác 0 và lần lượt cả bốn G anchors để bao phủ các nhánh. Chỉ dùng
ở run smoke riêng. Chương trình kiểm tra logs/counters/đủ rank files; kiểm tra
tensor ownership, teacher frozen, EMA timing và replay số học vẫn cần tests cùng
GPU inspection. Resume phải khôi phục counters, prompt cursor, RNG và local AdamW
partition. Cần cùng Torch build, rank topology, batch, dataset hash và config khoa
học. Kiểm tra `tests/test_h100_runtime.py` trước khi dùng rank-local resume dài.

Đọc peak VRAM từng rank, throughput, host RAM và tốc độ checkpoint. Batch 1 là
mức bắt đầu bảo thủ. Nếu muốn thử batch 2/GPU, dùng `PER_DEVICE_BATCH_SIZE=2` và
`RUN_ROOT` mới, chạy lại smoke. Chốt một batch rồi giữ nguyên cho tất cả candidate,
control và continuation. Code sẽ từ chối resume với batch khác. Nếu batch 1 vẫn
OOM, dừng và ghi vị trí/peak thực tế để điều chỉnh memory backend; không giảm
resolution hoặc bật FSDP/accumulation chưa được triển khai.

## Hai đường chạy, chọn theo ngân sách

Đường ngắn, mặc định: evaluate initialization → smoke/resume → pilot 1500 G →
evaluate pilot → nếu ổn, resume đến 5000 tổng G → full final evaluation.

```bash
bash scripts/run_h100.sh pilot cd-default
# Hoàn tất pilot evaluation trước khi quyết định tiếp tục.
bash scripts/run_h100.sh main cd-default
```

Đường sweep: ba candidate cùng seed/data/batch/init, chỉ đổi CD maximum:

```bash
bash scripts/run_h100.sh screen cd-low       # CD 0.03, 300 G
bash scripts/run_h100.sh screen cd-default   # CD 0.10, 300 G
bash scripts/run_h100.sh screen cd-high      # CD 0.30, 300 G
# Ví dụ nếu cd-default được chọn từ evidence:
bash scripts/run_h100.sh pilot cd-default   # resume 300 -> 1500, giữ nguyên ramp
bash scripts/run_h100.sh baseline           # beta=0, CD=0, 1500 G đối chứng
bash scripts/run_h100.sh main cd-default    # resume 1500 -> 5000 nếu đáng tiếp tục
```

Không chọn trước winner chỉ vì tên default. Sweep 900 G tổng chỉ sàng lọc CD sớm:
ở 300 G beta mới 0.01, chưa thể kết luận anchor đầy đủ có tác dụng. Confirmation
cho winner thêm 1200 G đến 1500; control 1500 G. Tổng giai đoạn screen + confirm +
control là **3600 G / 18000 F** khi mọi bước thành công. Nếu tiếp tục winner đến
5000 tổng G, toàn bộ kế hoạch là **7100 G / 35500 F**, chưa tính smoke hay retries.
Không có ước lượng giờ chạy cho đến khi đo throughput.

Control là beta=0/CD=0 trong cùng adapter decoupled/prompt-only. So winner/control
ở 1500 G, cùng batch và compute-unit convention; baseline initialization ở k_G=0
cũng phải được đánh giá. Control này không phải reproduction official DMD2 có GAN.
Nếu cần tách tác dụng beta và CD sau đó, thêm từng ablation beta-only/CD-only bằng
run mới; chưa đưa chúng vào ngân sách mặc định.

Chọn candidate theo protocol viết xuống **trước khi xem điểm**: bỏ run có
nonfinite/skip lặp, drift/collapse rõ hoặc metric thiếu provenance; so fixed
samples, corrected-F error/drift, CD-to-direct gradient ratio và metric validation
cùng prompts/seeds. CD MSE nhỏ không tự chứng minh ảnh tốt. Nếu không có bằng
chứng rõ, giữ CD 0.1 để confirm. Không đổi LR/beta/EMA cùng sweep đầu này.

Dùng khoảng 256–1000 prompts validation riêng, không trùng train và tránh dùng
COCO10K benchmark để tune. Log hash/mapping/seed/count cho subset. Nếu chỉ có
COCO subset sẵn, ghi rõ đã dùng nó để lựa chọn và mức thiên lệch; không trình bày
final benchmark như holdout hoàn toàn. Full COCO10K chỉ cho initialization/final
sau khi đã chốt cấu hình. Pilot FID ít mẫu chỉ có giá trị so nội bộ cùng protocol.

## Xem training và nối metrics đang có

```bash
tensorboard --logdir "$RUN_ROOT" --port 6006
```

- `fake_by_kF`: mixed loss, corrected error theo time bucket, số mẫu thực tế,
  grad norm/clip, LR, success/attempt, beta và thời gian step.
- `generator_by_kG`: direct/CD loss, DM/CA norm, gradient ratio trong output space,
  CD active/anchor, endpoint stats, LR/clip/success, thời gian update EMA CPU và logical forward counts.
- `probe_by_kG`: noise/prompt cố định; critic tracking, drift G/F và CD residual.
- `performance_by_kG`: global batch, thời gian cycle, VRAM allocated/reserved/peak và peak RSS của từng process (không phải RAM toàn node).
- `fixed_samples`: G online và EMA trên cùng 8 prompts/seeds.
- `evaluation`: chỉ điểm từ report đánh giá thật, kèm provenance dạng text.

TensorBoard chỉ ghi rank 0; loss là mean toàn ranks, bucket error weighted theo số
mẫu, counts/forward counts là sum, VRAM/timing là max. JSONL giữ giá trị từng rank.
F plots dùng k_F, G/probe/eval dùng k_G. Bucket không có mẫu được bỏ qua trong
TensorBoard và giữ null trong JSONL. Không lưu histogram weights/gradients hay
ảnh từng batch. Fixed sample generation bảo toàn RNG train; VAE được giải phóng
sau event. Giá trị loss proxy không phải thước đo trực tiếp chất lượng ảnh.

Tái dùng evaluator của bên nhận qua `--metric-plugin package.module:function`
như [EVALUATION.md](EVALUATION.md), hoặc tạo report cùng schema từ evaluator đã có.
Không tự suy đoán scale HPS/ImageReward. Import report vào dashboard cùng run:

```bash
fdmd log-evaluation --report reports/pilot/evaluation_manifest.json \
  --run-dir "$RUN_ROOT/cd-default" --generator-updates 1500
```

Report từ code repo là `evaluation_manifest.json` (xem EVALUATION.md). Kết quả trả về cần FID, CLIP-S, ImageReward, HPSv2.1/HPSv3 nào thực
sự có, thang điểm, init→pilot→final delta và weight choice. Có thể vắng một metric;
không điền số giả. Chốt một lựa chọn G/EMA bằng validation và dùng nhất quán cho
final10K; samples/evaluator chạy bằng một GPU, tách khỏi training để tránh OOM.

## Chỉ giữ dữ liệu cần thiết

Checkpoint mỗi 500 G giúp giảm I/O full optimizer state. `checkpoint_keep_last: 1` chỉ prune checkpoint periodic/final do schema 2 tạo,
sau khi save mới hoàn tất và có checksum. `.pin`, failure/warmup checkpoints,
checkpoint schema cũ, assets/source weights và exports được giữ. Để bảo toàn một
candidate cần rollback, tạo file `.pin` trong đúng thư mục checkpoint trước run
kế tiếp. Không xóa auto một selected/best model.

Sau khi evaluate pilot, export **EMA hoặc G đã chọn** trước khi main thay latest:

```bash
# Điền checkpoint thật từ latest.json; export không dùng placeholder này để chạy.
fdmd export --checkpoint "$SELECTED_CHECKPOINT" --weights ema \
  --output runs/selected/pilot-ema.safetensors
```

Chỉ giữ một selected pilot export nếu cần so final, một latest resume của winner
và final selected export; logs/TensorBoard/config/reports nhỏ giữ đủ. Sau khi chọn
sweep và ghi bảng kết quả, bên nhận có thể xóa checkpoint thư mục của candidate
thua/control nếu không cần replay; giữ JSONL, TensorBoard và manifests kết quả.
Không tự prune chéo run vì chưa biết candidate nào đã được chọn. `.pin` cũng cần
được rà soát để tránh giữ quá nhiều full optimizer states.

Gửi lại commit/config hash, env freeze, asset/metric hashes, initial/screen/pilot/
final reports, logs/TensorBoard, selected weights và checklist GPU đã thực hiện.
Mục tiêu quan trọng là xác định continuation có cải thiện chất lượng/ổn định hay
không; nếu pilot hỏng hoặc final không cải thiện, báo đúng điểm và failure modes.
