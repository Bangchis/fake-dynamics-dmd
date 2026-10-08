# Pilot nhanh: BF16 online + LR 2e-6 trên 4 H100

Yêu cầu ngày 2026-10-09: giảm thời gian lấy kết quả để kiểm tra phương pháp, dùng
BF16 và tăng LR. Đây là kế hoạch mới thay run dài batch128; chưa đo tốc độ, khả năng
vừa VRAM hay chất lượng. Các thay đổi khoa học được ghi trong DECISIONS.md.

## Dùng đúng backend bên nhận

Theo audit bên nhận, họ đã có G/F online BF16 CUDA, master weights/AdamW moments
FP32, và offload CPU khi optimizer idle. Code đó đang local; tác giả chưa nhận
patch hoặc các file mới. Backend public vẫn giữ G/F FP32 với BF16 autocast.
Đặt mixed_precision=bf16 trong backend public không tự tạo BF16 online weights.

Overlay `configs/recipient_fast_pilot_overrides.yaml` dành cho **bản BF16 online với
FP32 master của bên nhận**, không phải config standalone cho backend public.
Script tạo config chỉ đọc/ghi YAML, không import Torch hoặc chạy model; nó yêu cầu
base config đã có train_weight_dtype=bfloat16/bf16, giữ đường dẫn asset và các
field backend local, rồi thay các thông số pilot. Nó không chứng minh backend
có master weights đúng chỉ từ giá trị field đó.

Trước pilot, sửa EMA để lấy từ **FP32 master G**, không lấy từ online G đã cast
BF16. Giữ master G/F, AdamW moments và EMA arithmetic/checkpoint FP32. Giữ strict
import/optimizer ownership và cơ chế skip đồng bộ. Kiểm tra save/resume/export
trên đúng 4 GPU và ghi delta master/timing từ máy thật. Không cast toàn bộ
optimizer state sang BF16 chỉ để tăng physical batch.

## Config được chọn

| Mục | Giá trị |
|---|---|
| Online G/F, teacher, EMA target forward | BF16; master/AdamW/EMA arithmetic FP32 |
| Resolution / G sampler | 1024; 999/749/499/249, stochastic re-noising, conditional-only |
| G/F LR | 2e-6 / 2e-6 constant; gấp 4 reference DMD2 5e-7 |
| AdamW / clip | betas .9/.999, decay .01, clip 10 |
| Batch mục tiêu | 8/GPU × 4 ranks × accumulation1 = global32 |
| Activation checkpointing | Bật khi thử batch8; đo throughput để chọn |
| F:G / budget | 5:1; 200 successful G / 1000 successful F |
| Beta | max .05; start G20; ramp80 → đầy đủ từ k_G=100 |
| CD / EMA | CD max .1/ramp100; EMA decay .99 |
| Logs | JSONL từng update/rank, TensorBoard, performance10G/probe25G |
| Ảnh / checkpoint | 8 fixed prompts × G/EMA mỗi50G; checkpoint100G, giữ latest1 |

Đây là run mới từ cặp DMD2 G/F gốc, optimizer/counters mới, directory riêng.
Không resume run cũ để đổi LR/batch/ramp. Ramps đo bằng successful G updates,
không theo microbatch hoặc F counter. G update khi k_G=100 bắt đầu dùng beta max;
log sau update G100 có thể là weight dùng ở k_G=99, cần đọc đúng field.

Batch8 chưa được xác minh vừa VRAM. Audit beta0/batch4 không chứng minh điều đó.
Đo một run ngắn riêng với beta=.05/CD=.1 và bốn anchor, cùng physical batch/precision
backend dự định dùng. Giữ overrides/debug_anchor_cycle null/false trong quality run.

## Tạo config trên máy bên nhận

Lấy base từ resolved run manifest hoặc config YAML local có đầy đủ asset paths.
Không dùng `configs/h100_2x80.yaml` làm base cho script này vì backend BF16 online
của họ chưa được merge. Ví dụ từ repo root của bên nhận, thay đường dẫn BASE thật:

```bash
python scripts/make_fast_pilot_config.py \
  --base /absolute/path/to/current/run_manifest.json \
  --output configs/local-fast-b32.yaml \
  --run-dir runs/fast-pilot-b32-lr2e-6 \
  --mode b32-noaccum --activation-checkpointing on

# Chỉ chạy trên bản code local đã có BF16 online + FP32 master, EMA-from-master.
# RunAI cần cấp 4 GPU cho pod; torchrun không tự tạo thêm GPU.
torchrun --standalone --nproc_per_node=4 -m fake_dynamics.cli --debug train \
  --config configs/local-fast-b32.yaml
```

Không dùng launcher `scripts/run_h100.sh` cho pilot này: launcher cũ hardcode
2 ranks/global128 và các budget dài. Helper từ chối overwrite config hoặc directory
run cũ. Kiểm tra resolved manifest và LR thực tế của **cả hai optimizers** là 2e-6.
Không cần pull đè working tree local đang dirty; có thể lấy riêng overlay/script/doc
và áp dụng vào code của họ. Cần lưu patch + file mới để tác giả review/merge sau.

Nếu batch8 OOM hoặc checkpointing làm chậm hơn, chọn một mode mới trước quality run:

| --mode | Batch/GPU | Accumulation | Global | Ý nghĩa |
|---|---:|---:|---:|---|
| b32-noaccum | 8 | 1 | 32 | Mục tiêu bỏ accumulation; chưa đo fit |
| b32-accum2 | 4 | 2 | 32 | Đo cùng global32; có thể chạy nhanh hơn nếu tránh recomputation |
| b16-noaccum | 4 | 1 | 16 | Pilot nhẹ hơn, khác global batch; ghi rõ trong report |

Mode đổi cần config/run-dir mới. Có thể dùng --activation-checkpointing off khi đã
đo vừa VRAM ở mode đó. Nếu OOM trong optimizer step, kiểm tra lượng master/moments
resident/sharded thay vì chỉ bật activation checkpointing. Nếu OOM trong forward/
backward, kiểm tra attention/activation memory. Đo cả VRAM và wall time, không đo
chỉ GPU kernel time; CPU↔GPU copy/NCCL/checkpoint cũng nằm trong budget.

Giảm 128→32 giảm samples/compute mỗi optimizer window. Bỏ accum2 để tăng physical
4→8 giữ global32 chỉ đổi cách thực thi; không tự bỏ compute. LR lớn hơn có mục tiêu
làm updates mạnh hơn, không tự giảm thời gian mỗi cycle hay bảo đảm hội tụ tốt.

## Quyết định dựa trên kết quả

1. Chuẩn bị COCO10K + evaluator đồng thời với sửa code; đo initialization bằng
   cùng sampler/precision/weight choice như checkpoint mới. Cache metric sẵn có
   được tái dùng nếu version/checkpoint/scale/provenance đã xác minh.
2. Pilot200G tạo checkpoint đầu tiên. Dùng tập prompts riêng để xem ảnh/metric sớm;
   không tune trên COCO10K final. Đo full10K cho checkpoint đã chọn theo EVALUATION.md.
3. Có cải thiện init thì chạy control beta=0/CD=0 với cùng LR, mode, init, prompts,
   seed, số successful G/F updates và protocol. Dùng cùng code EMA/master đã sửa.
   Gain so control là bằng chứng cho các thành phần mới trong setting pilot.
4. Nếu đáng tiếp tục, mở rộng candidate/control đến400 tổngG với các science fields
   giữ nguyên. Pilot với ramp nhanh không chứng minh setting start100/ramp1000.
5. Báo riêng init, candidate, control và paper references. So chất lượng với paper
   cần protocol tương thích; khác batch/LR/ramp và inherited DMD2 training phải rõ.
   Không gọi đây là equal-budget reproduction chỉ vì điểm có thể đặt cạnh bảng.

Giữ config/manifest/TensorBoard/logs, latest resume và selected export/report.
Chưa có measured FID/CLIP/ImageReward/HPS, bằng chứng fit hoặc ETA cho pilot này.
