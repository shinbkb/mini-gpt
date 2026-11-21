# MiniGPT-Sys

Dự án xây dựng Decoder-Only Transformer từ đầu bằng PyTorch.
Hiện đã triển khai M0, M1 và M2: môi trường, BPE/TinyStories và Dense GPT baseline.
Kế hoạch: [IMPLEMENTATION_PLAN_VI.md](IMPLEMENTATION_PLAN_VI.md).

## Cấu trúc

- `configs/`: cấu hình mục tiêu Dense và MoE.
- `src/`: tokenizer BPE, TinyStories Dataset, Dense GPT, trainer, KV cache và generation.
- `tests/`: CPU smoke, BPE, dữ liệu, model, causal mask, overfit và checkpoint.
- `notebooks/00_setup.ipynb` và `01_tokenizer.ipynb`: M0/M1 Kaggle.
- `notebooks/02_dense_gpt.ipynb`: huấn luyện/generate Dense GPT trên Kaggle.
- `experiments/`, `reports/`, `artifacts/`: thí nghiệm, báo cáo, output.

M3 bổ sung KV cache và benchmark decode. M4 bổ sung MoE Top-2 và benchmark Dense/MoE.

## Ubuntu — cài đặt và kiểm tra CPU

Yêu cầu Python >=3.11. Từ thư mục gốc repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -c "import src; print('import src: OK')"
python -m src.environment --seed 42
python -m pytest -q
```

Nếu `.venv` đã có, chỉ activate và kiểm tra dependencies trước khi cài thêm.
GPU: cài bản PyTorch phù hợp driver/CUDA thay cho index CPU.
M0 không yêu cầu GPU. Lệnh environment in Python, phiên bản thư viện, seed,
CUDA build/availability, tên GPU và tổng VRAM nếu có. Seed áp dụng cho Python
và PyTorch, không cam kết kết quả bitwise giống nhau giữa mọi thiết bị.

## Kaggle

1. Commit và push M0 lên GitHub; lấy hash đầy đủ bằng `git rev-parse HEAD`.
2. Mở `notebooks/00_setup.ipynb`, điền `COMMIT_SHA` chứa M0.
3. Bật Internet nếu clone online. Notebook clone repository rồi checkout đúng
   commit, xác minh HEAD và chạy kiểm tra môi trường/tests qua module `src`.
4. GPU là tùy chọn trong M0; có thể dùng CPU. Không ghi token truy cập vào notebook.

Khi không có Internet: đóng gói source của commit đã chọn thành Kaggle Input,
đặt `OFFLINE_SOURCE` và `SOURCE_COMMIT` trong notebook. Dependencies phải có sẵn
hoặc cài trước từ wheelhouse local. Notebook copy source vào `/kaggle/working`,
không tải dependencies ở chế độ offline. Commit của gói offline do người đóng gói
khai báo; không coi đó là xác minh mật mã nội dung source.

Lưu output dưới `/kaggle/working/` và lưu Notebook Output/download trước khi kết
thúc phiên. `.gitignore` bỏ qua secrets, dữ liệu, cache và checkpoints; không
bỏ qua mã nguồn `src/data/`.

## M1 — TinyStories và byte-level BPE

M1 thêm `ByteLevelBPE`, reader TinyStories, `CausalLMDataset`, CLI chuẩn bị dữ
liệu, tests và notebook `01_tokenizer.ipynb`. Chưa có model/training cho M2 trở đi.

Cài requirements trong virtualenv (PyTorch CPU/CUDA cài theo máy):

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Chuẩn bị dữ liệu online trên máy có Internet:

```bash
python -m src.data.prepare \
  --train-limit 20000 --validation-limit 1000 \
  --vocab-size 8000 --block-size 512 --require-full-vocab \
  --revision DATASET_COMMIT_SHA --output-dir artifacts
```

Tokenizer chỉ học từ train. Validation được encode riêng. Dùng hai file local
`--train-file train.jsonl --validation-file validation.jsonl` khi offline;
JSONL cần trường `text`, TXT phân tách truyện bằng dòng `<|endoftext|>`.
Không truyền cùng một file làm cả hai split. Output artifacts bị Git bỏ qua.
Corpus nhỏ có thể không tạo đủ 8.000 token; `--require-full-vocab` sẽ báo lỗi.

`notebooks/01_tokenizer.ipynb` clone và checkout commit đầy đủ, chạy tests rồi gọi
CLI từ `src`. Sau khi push code M1, điền `COMMIT_SHA`; để tái lập dữ liệu online,
điền `DATASET_REVISION`. Notebook không chứa lại thuật toán tokenizer.


## M4 — Mixture of Experts

`configs/moe.yaml` cấu hình 4 expert, Top-2 routing và auxiliary loss
`0.01 * E * sum_i(f_i * P_i)`. `train.py` dùng cùng TinyStories token cache
với Dense, huấn luyện MoE từ đầu, lưu `runs/moe/{best,last}.pt` và metrics có
validation loss/PPL, throughput, VRAM, total/activated parameters, expert token
counts, utilization và router entropy.

CPU smoke (dùng token cache M1 và giảm số bước):

```bash
python train.py --config configs/moe.yaml --max-steps 2 --device cpu
```

So sánh Dense và MoE trên cùng validation split, context, batch, precision và
device; benchmark ghi loss/PPL, validation tokens/s, peak VRAM, parameter counts
và routing distribution:

```bash
python -m experiments.benchmark_moe \
  --dense-checkpoint runs/dense/best.pt \
  --moe-checkpoint runs/moe/best.pt \
  --validation-tokens artifacts/validation_tokens.pt \
  --precision fp16 --device auto --output reports/benchmark_moe.json
```

Notebook `04_moe.ipynb` điều phối Kaggle. Gắn M1 artifacts làm Input, đặt đúng
đường dẫn trong cell đầu, cập nhật SHA đầy đủ sau khi push. Notebook huấn luyện
Dense baseline và MoE từ đầu với cùng số bước × batch × context × gradient
accumulation, sau đó benchmark và Save Version; không kết luận MoE tốt hơn Dense
nếu số liệu không ủng hộ.

## M2 — Dense GPT

Bắt đầu bằng tiny config trên cache M1:

```bash
python train.py --config configs/tiny.yaml --max-steps 100 --device auto
python -m src.inference.generate --checkpoint runs/tiny/best.pt \
  --tokenizer artifacts/tokenizer.json --prompt "Once upon a time"
python evaluate.py --checkpoint runs/tiny/best.pt --tokens artifacts/validation_tokens.pt

# So sánh cache/no-cache. Context dài hơn block_size của checkpoint sẽ tự bỏ qua.
python -m experiments.benchmark_kv --checkpoint runs/dense/best.pt \
  --contexts 128 512 1024 2048 --precision fp16 --device auto
```

Trainer ghi validation loss/perplexity, tokens/s, peak VRAM và lưu `last.pt`,
`best.pt`, `metrics.jsonl`. Resume bằng `--resume runs/tiny/last.pt`.
`--overfit-batch` kiểm tra khả năng giảm loss trên một batch cố định.
Mô hình proposal-scale dùng `configs/dense.yaml`; chạy sau khi xác nhận tiny config.

Trên Kaggle: commit/push M2, điền hash vào `notebooks/02_dense_gpt.ipynb`,
đính kèm M1 artifacts đã tải lên như Kaggle Dataset và đặt `M1_ARTIFACTS_INPUT`
thành thư mục input. Chọn Internet để clone/cài dependencies; GPU tùy chọn.
Notebook mặc định chạy tiny 100 steps, sau đó sinh mẫu văn bản. Dùng Save Version
để lưu metrics và checkpoints.
