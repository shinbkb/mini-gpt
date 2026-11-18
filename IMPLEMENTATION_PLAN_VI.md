# MiniGPT-Sys — Kế hoạch triển khai dành cho Codex

> **Đề tài:** Xây dựng và tối ưu mô hình ngôn ngữ Decoder-Only Transformer từ đầu bằng PyTorch.  
> **Hướng:** A — Systems: tối ưu suy luận, kiến trúc MoE và mở rộng ngữ cảnh.  
> **Môi trường:** Ubuntu + VS Code/Cursor để phát triển; Kaggle Notebook (GPU T4 16 GB nếu khả dụng) để huấn luyện và benchmark.  
> **Dữ liệu:** `roneneldan/TinyStories`; chỉ dùng train để học tokenizer và huấn luyện mô hình, giữ validation độc lập.  
> **Thời lượng:** 4 tuần.  
> **Căn cứ:** Proposal MiniGPT-Sys (`llm.pdf`, 5 trang). Các lưu ý triển khai và phương án giảm phạm vi bên dưới là đề xuất kỹ thuật bổ sung.

## 0. Quy tắc bắt buộc dành cho Codex

Bạn là trợ lý lập trình làm việc trong repository này. Hãy:

1. **Trao đổi, giải thích và báo cáo tiến độ bằng tiếng Việt**; giữ tên file, class, function, biến, lệnh và thuật ngữ kỹ thuật bằng tiếng Anh khi phù hợp.
2. Trước khi sửa code: kiểm tra repository hiện có, trình bày kế hoạch ngắn gọn, nêu các giả định.
3. **Chỉ triển khai milestone được người dùng giao.** Không tự ý làm các milestone sau. Sau mỗi milestone: chạy test, báo cáo file đã sửa, lệnh chạy, kết quả, vấn đề còn tồn tại rồi chờ xác nhận.
4. Tự lập trình **byte-level BPE, attention, RoPE, RMSNorm, Transformer, KV Cache, MoE** bằng Python/PyTorch. Không thay bằng mô hình hoặc tokenizer dựng sẵn từ `transformers`, `tokenizers`, `tiktoken` hay các package tương tự. Có thể dùng `datasets` để tải dữ liệu, `torch` cho tensor/autograd, `pytest` để kiểm thử.
5. Không tải pretrained model, không sử dụng validation để huấn luyện hoặc lựa chọn merge BPE. Không đưa dataset, checkpoint, tokenized cache dung lượng lớn lên Git.
6. Ưu tiên code dễ đọc, cấu hình rõ ràng, có type hints/docstrings, seed, kiểm tra lỗi, unit tests; không tối ưu sớm trước khi có kết quả đúng.
7. Không khẳng định đạt tốc độ, perplexity hay độ chính xác khi chưa có số liệu thực đo. Ghi rõ thiết bị, batch, context length, precision, số lần lặp và commit khi benchmark.
8. Nếu GPU Kaggle không khả dụng, vẫn hoàn thiện smoke test CPU với cấu hình nhỏ. Không giả định Kaggle có Internet hoặc GPU mọi lúc.
9. Tôn trọng ngân sách 4 tuần: ưu tiên baseline + KV Cache + MoE; RoPE Scaling và INT8 triển khai sau khi các mốc cốt lõi ổn định.

## 1. Mục tiêu và sản phẩm bàn giao

### 1.1 Mục tiêu khoa học

- Xây dựng một mô hình autoregressive Decoder-Only Transformer có khả năng sinh văn bản từ TinyStories.
- Đánh giá hiệu quả **KV Cache** đối với độ trễ và tốc độ sinh token.
- So sánh **Dense GPT** và **MoE GPT** có Top-2 Routing, auxiliary load balancing loss.
- Thử mở rộng context từ 512 lên 1024–2048 bằng RoPE Scaling và kiểm tra khả năng truy xuất thông tin dài.
- Thử post-training INT8 quantization, đo đánh đổi giữa dung lượng, tốc độ, VRAM và chất lượng.

### 1.2 Cấu hình mục tiêu từ proposal

| Siêu tham số | Giá trị |
|---|---:|
| `n_layers` | 6 |
| `d_model` | 384 |
| `n_heads` | 6 |
| `head_dim` | 64 |
| `d_ff` | 1536 |
| `vocab_size` | 8000 |
| `block_size` | 512 |
| Context mở rộng | 1024–2048 |
| `n_experts` | 4 |
| `top_k` | 2 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning rate ban đầu | 3e-4 |
| Batch size ban đầu | 8, điều chỉnh theo VRAM |
| Precision | FP16/mixed precision nếu phù hợp |

Proposal dự kiến khoảng **13,7 triệu tham số Dense** và **35 triệu tham số MoE**. Đây là ước lượng, **phải xác nhận bằng `sum(p.numel() for p in model.parameters())`**; đặc biệt nêu rõ có weight tying, bias, loại FFN và cách đếm activated parameters hay không.

### 1.3 Sản phẩm cuối

- Repository có cấu trúc module, test, config, hướng dẫn chạy và lịch sử Git.
- Tokenizer BPE tự viết, vocabulary/merge rules có thể lưu và tải lại.
- Dense GPT và MoE GPT, script train/evaluate/generate, checkpoint hợp lệ.
- KV Cache và benchmark có thể tái lập.
- Bảng/biểu đồ Dense vs MoE, KV Cache on/off; thêm Long Context và INT8 nếu hoàn thành.
- Báo cáo kỹ thuật/khoa học và demo inference.

## 2. Cấu trúc repository

```text
MiniGPT-Sys/
├── configs/
│   ├── dense.yaml
│   └── moe.yaml
├── src/
│   ├── __init__.py
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   └── bpe.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rmsnorm.py
│   │   ├── rope.py
│   │   ├── attention.py
│   │   ├── transformer.py
│   │   ├── moe.py
│   │   └── gpt.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── kv_cache.py
│   │   └── generate.py
│   └── training/
│       ├── __init__.py
│       └── trainer.py
├── experiments/
│   ├── benchmark_kv.py
│   ├── benchmark_moe.py
│   ├── benchmark_context.py
│   └── benchmark_int8.py
├── tests/
├── notebooks/
│   ├── 01_tokenizer.ipynb
│   ├── 02_dense_gpt.ipynb
│   ├── 03_kv_cache.ipynb
│   ├── 04_moe.ipynb
│   └── main_pipeline.ipynb
├── reports/
├── artifacts/               # Không commit dữ liệu/checkpoint lớn
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── README.md
└── IMPLEMENTATION_PLAN.md
```

Notebook là **lớp điều phối**: clone đúng commit GitHub, import module từ `src`, chạy train/benchmark, lưu output. **Không nhân đôi thuật toán trong notebook.** Có thể thêm `pyproject.toml` để cài đặt package editable.

## 3. Lộ trình 4 tuần và các milestone

| Tuần | Milestone | Kết quả tối thiểu |
|---|---|---|
| 1 | M0 + M1 | Repo chạy được; TinyStories; BPE 8k; test round-trip |
| 2 | M2 + M3 | Dense GPT huấn luyện được; generation; KV Cache đúng và có benchmark |
| 3 | M4 + bắt đầu M5 | MoE Top-2; auxiliary loss; so sánh có kiểm soát; thử long context |
| 4 | M5/M6 + M7 | Hoàn thiện benchmark, error analysis, báo cáo, demo |

### M0 — Khởi tạo repository và môi trường

**Cần làm:**

- Tạo cấu trúc thư mục, `__init__.py`, `README.md`, `.gitignore`, `requirements.txt`, YAML configs.
- Kiểm tra Python, PyTorch, CUDA; in thiết bị, VRAM nếu có, version thư viện và seed.
- Thiết lập `pytest`; có ít nhất một smoke test chạy trên CPU.
- Cấu hình Git bỏ qua `.env`, dữ liệu, cache, checkpoint và file output lớn.
- Tạo ví dụ notebook Kaggle clone repository bằng commit hash cố định.

**Nghiệm thu:** `python -m pytest -q` chạy được; import `src` không lỗi; README hướng dẫn cài đặt Ubuntu/Kaggle.

### M1 — TinyStories và Custom Byte-Level BPE Tokenizer (LÀM ĐẦU TIÊN)

**File chính:** `src/tokenizer/bpe.py`, `src/data/dataset.py`, `tests/test_bpe.py`, `notebooks/01_tokenizer.ipynb`.

**Yêu cầu:**

1. Tải `roneneldan/TinyStories` bằng Hugging Face `datasets` khi Internet khả dụng; hỗ trợ đường dẫn dữ liệu local/Kaggle Input nếu không có Internet.
2. Dùng **train split** để huấn luyện BPE; giữ validation độc lập. Khởi đầu với mẫu khoảng 20.000 truyện, mở rộng sau khi kiểm thử hiệu năng.
3. Tự viết **byte-level BPE**: khởi tạo 256 byte token, đếm cặp liền kề, chọn cặp theo tần suất và quy tắc tie-break xác định, merge theo thứ tự, tạo token mới; dành ID cho special tokens (ví dụ `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|unk|>` nếu cần). Bảo đảm tổng vocabulary theo config là 8.000, bao gồm special tokens.
4. Thiết kế API: `train(texts, vocab_size)`, `encode(text, add_bos=False, add_eos=False)`, `decode(ids, skip_special_tokens=False)`, `save(path)`, `load(path)`.
5. Lưu đầy đủ bytes/vocab, merge ranks, special tokens, version và config dưới định dạng JSON an toàn (mã hóa bytes bằng danh sách số hoặc base64).
6. Không lower-case, không xóa dấu câu, không normalize khiến dữ liệu không thể khôi phục nếu không có lý do nghiên cứu rõ ràng.
7. Dùng tokenizer để chuyển văn bản thành token IDs. Xây dựng dữ liệu Causal LM: `x=tokens[i:i+block_size]`, `y=tokens[i+1:i+block_size+1]`; tránh nối tùy tiện các truyện không có EOS phân cách.
8. Cần có chiến lược tối ưu thuật toán vì BPE thuần Python quét lại toàn bộ corpus 8.000 lần có thể rất chậm. Bắt đầu từ tập nhỏ, đo thời gian; có thể cải tiến bằng bộ đếm cặp, cập nhật cục bộ hoặc biểu diễn pre-tokenized mà vẫn tự cài thuật toán.

**Unit tests bắt buộc:**

- `decode(encode(text)) == text` cho ASCII, Unicode/tiếng Việt, dấu câu, xuống dòng, chuỗi rỗng, emoji (không chèn special tokens).
- Cùng corpus + seed/tie-break cho kết quả xác định.
- Save/load không làm thay đổi encode/decode.
- Vocab không vượt config, special IDs không trùng, token IDs luôn hợp lệ.
- Test merge trên corpus toy có kết quả kỳ vọng; test các trường hợp thiếu file/cấu hình không hợp lệ.
- Test Dataset trả về tensor `x`, `y` đúng shape/dtype và shift một token.

**Nghiệm thu:** chạy notebook Kaggle tạo tokenizer 8.000 token; xuất `artifacts/tokenizer.json`; có số liệu thời gian train/encode và kết quả test; chưa cần GPU.

### M2 — Dense GPT baseline

**File chính:** `src/models/rmsnorm.py`, `rope.py`, `attention.py`, `transformer.py`, `gpt.py`, `src/training/trainer.py`, `train.py`, `configs/dense.yaml`.

**Yêu cầu:**

- Tự cài RMSNorm, RoPE, causal multi-head self-attention, residual connections, pre-norm decoder blocks, FFN, final RMSNorm và LM head.
- Tạo causal mask đúng; không cho token nhìn tương lai. Xác định cách áp dụng RoPE vào Q/K, kiểm tra vị trí token và tensor shapes.
- Training với cross-entropy next-token prediction, AdamW, gradient clipping, learning-rate schedule nếu dùng, mixed precision phù hợp, checkpoint gồm model/optimizer/scaler/config/step/RNG state.
- Tách train/validation, log loss, perplexity (`exp(loss)` khi hữu hạn), tokens/s, VRAM.
- Bắt đầu bằng **tiny config** (ví dụ 2 layers, d_model 128, context 64) để overfit một batch nhỏ, rồi mới tăng lên cấu hình 6 layers, d_model 384, context 512.
- Có script generate với temperature, top-k và seed; dừng theo EOS hoặc max_new_tokens.

**Unit tests:** causal masking (thay token tương lai không đổi logits vị trí trước), shape, finite loss, backward gradients, overfit tiny batch, checkpoint reload.

**Nghiệm thu:** Dense GPT train/infer thành công; validation loss được ghi lại; sinh được văn bản; số tham số đếm trực tiếp.

### M3 — KV Cache và profiling

**File chính:** `src/inference/kv_cache.py`, `src/inference/generate.py`, `experiments/benchmark_kv.py`, `tests/test_kv_cache.py`.

**Yêu cầu:**

- Cache K/V **theo từng attention layer**, theo batch/head/sequence/head_dim, bảo đảm đúng dtype/device và vị trí RoPE.
- Phân biệt **prefill** (xử lý prompt) và **decode** (sinh token mới); hỗ trợ `use_cache=True/False`.
- Đảm bảo logits cache và no-cache tương đương trong sai số số học hợp lý, ở chế độ eval/dropout off.
- Benchmark context length `{128, 512, 1024, 2048}` **chỉ với các cấu hình thật sự hỗ trợ**. Không ép model context 512 chạy 2048 nếu chưa triển khai mở rộng.
- Đo first-token latency (định nghĩa rõ có gồm prefill hay không), next-token latency, throughput tokens/s, peak GPU VRAM; warmup và đồng bộ CUDA trước/sau khi đo.
- Giữ nguyên model, prompt, seed, precision, batch, số token sinh giữa hai chế độ.

**Nghiệm thu:** test correctness qua; có bảng số liệu và biểu đồ cache on/off; không đưa ra speedup giả định.

### M4 — Mixture of Experts (MoE)

**File chính:** `src/models/moe.py`, `configs/moe.yaml`, `experiments/benchmark_moe.py`, `tests/test_moe.py`.

**Yêu cầu:**

- Thay Dense FFN bằng 4 experts; router tuyến tính dự đoán expert logits, chọn Top-2 và softmax chuẩn hóa trọng số của 2 experts được chọn.
- Tự cài dispatch/gather cho token; đảm bảo gradient đến router và selected experts; xử lý batch/sequence shapes.
- Thêm auxiliary load-balancing loss, ghi rõ công thức và hệ số, theo dõi expert utilization, routing distribution, token counts, entropy hoặc chỉ số cân bằng phù hợp.
- Huấn luyện MoE từ đầu trên cùng train/validation TinyStories; không tự tuyên bố MoE tốt hơn Dense nếu thực nghiệm không ủng hộ.
- So sánh có kiểm soát: số training tokens, context, tokenizer, optimizer, precision, GPU, batch và ngân sách tính toán. Báo cáo **total parameters**, **activated parameters** và compute thực tế.

**Unit tests:** đúng Top-2, trọng số routing tổng bằng 1, gradients hợp lệ, output shapes, finite auxiliary loss, save/load.

**Nghiệm thu:** MoE train/infer được; có loss/PPL, throughput, VRAM, biểu đồ phân bố expert và phân tích trade-off.

### M5 — Mở rộng ngữ cảnh bằng RoPE Scaling (ưu tiên sau M4)

**File chính:** `src/models/rope.py`, `experiments/benchmark_context.py`, `tests/test_rope.py`.

**Yêu cầu:**

- Thử linear interpolation hoặc base-frequency scaling từ 512 lên 1024/2048; ghi chính xác công thức và config.
- Phân biệt **khả năng chạy chuỗi dài** với **khả năng sử dụng thông tin dài**. Nếu cần, thực hiện continued training/fine-tuning với chuỗi dài.
- Xây dựng Needle-in-a-Haystack benchmark với thông tin mục tiêu ở đầu/giữa/cuối; dùng các prompt và tiêu chí chấm điểm rõ ràng, lưu kết quả theo context length và vị trí needle.
- Kiểm thử cache với vị trí token dài, tránh sai RoPE position offset.

**Nghiệm thu:** mô hình xử lý được độ dài đã công bố, không OOM trong điều kiện báo cáo; có bảng NIAH accuracy và phân tích hạn chế.

### M6 — INT8 post-training quantization (ưu tiên sau M4)

**File chính:** `experiments/benchmark_int8.py` và module hỗ trợ nếu cần.

**Yêu cầu:**

- Triển khai/đánh giá post-training weight quantization INT8 với phương pháp hỗ trợ GPU thực tế. Có thể dùng backend quantization cho phép, nhưng phải giải thích rõ phần tự viết và phần dùng thư viện.
- Đo kích thước file checkpoint, peak VRAM, latency, tokens/s, validation loss/PPL trước và sau lượng tử hóa.
- Không mặc định INT8 nhanh hơn FP16 trên T4: tốc độ phụ thuộc backend, kernel và workload.
- Nếu backend không hỗ trợ môi trường Kaggle, ghi rõ giới hạn và chỉ báo cáo phép đo thực hiện được, không giả lập kết quả.

**Nghiệm thu:** báo cáo tái lập được với baseline và INT8, hoặc mô tả trung thực lý do không thể chạy.

### M7 — Đánh giá, báo cáo, tái lập và demo

**Yêu cầu:**

- Hoàn thiện ít nhất 4 nhóm thí nghiệm trong proposal nếu tài nguyên cho phép: Dense vs MoE, KV Cache, RoPE Scaling/NIAH, INT8.
- Viết error analysis ít nhất 5 trường hợp mô hình sinh sai hoặc không truy xuất đúng thông tin; phân loại lỗi, giả thuyết nguyên nhân và giới hạn.
- Xuất bảng kết quả `.csv` và biểu đồ `.png`; ghi model config, commit hash, dataset version, tokenizer hash, GPU, seed, precision, thời gian train.
- Tạo README với quickstart, lệnh train/evaluate/benchmark và cách chạy Kaggle; thêm demo sinh văn bản terminal hoặc Gradio nếu kịp.
- Báo cáo 6–8 trang theo định dạng được CLB/giảng viên yêu cầu (nếu có), gồm mục tiêu, kiến trúc, dữ liệu, thiết lập, kết quả, ablation, hạn chế và hướng phát triển.

**Nghiệm thu:** người khác clone repo, cài dependencies, chạy tests, nạp checkpoint được cung cấp và tái lập benchmark theo hướng dẫn.

## 4. Giao thức đánh giá và chỉ số

| Nhóm | Chỉ số | Quy tắc |
|---|---|---|
| Language modeling | Validation loss, perplexity | Cùng validation split, tokenizer, cách xử lý EOS |
| Generation | Tokens/s, TTFT, TPOT | Tách prefill/decode; CUDA synchronize; warmup |
| Memory | Peak allocated/reserved VRAM | Reset peak stats trước mỗi lượt đo; ghi batch/context |
| MoE | Total/active params, expert load | Ghi rõ cách tính active params và routing |
| Long context | NIAH accuracy | Chia theo độ dài và vị trí needle; không dùng một ví dụ duy nhất |
| Quantization | Checkpoint size, PPL delta, speed | So sánh cùng input và điều kiện phần cứng |

Tất cả benchmark phải có ít nhất vài lần lặp, báo cáo trung bình và độ phân tán nếu khả thi; không so sánh các lần chạy với điều kiện khác nhau mà không ghi chú.

## 5. Tích hợp Kaggle với GitHub

**Nguyên tắc:** GitHub là nguồn mã chính; Kaggle chỉ clone **một commit cụ thể** và chạy notebook/script. Không sao chép toàn bộ class vào notebook.

Ví dụ cell Kaggle (thay `OWNER` và `COMMIT_SHA`):

```python
!git clone https://github.com/OWNER/MiniGPT-Sys.git /kaggle/working/MiniGPT-Sys
%cd /kaggle/working/MiniGPT-Sys
!git checkout COMMIT_SHA
!pip install -q -r requirements.txt
!python -m pytest -q
```

- Nếu repo private, dùng phương án xác thực an toàn; **không hardcode GitHub token** trong notebook.
- Nếu Kaggle không bật Internet, upload mã nguồn dạng Kaggle Dataset/Input hoặc dùng phiên bản notebook đã đóng gói source; ghi rõ commit nguồn.
- Lưu checkpoints/results trong `/kaggle/working/`; tải về hoặc lưu thành Kaggle Dataset/Notebook Output. Không dựa vào ổ đĩa tạm để lưu lâu dài.
- Notebook chính chỉ gọi CLI hoặc import module, ví dụ `python train.py --config configs/dense.yaml`; Codex cần tạo giao diện CLI thật trước khi dùng.

## 6. Định nghĩa hoàn thành (Definition of Done)

Một milestone chỉ được xem là hoàn thành khi:

1. Code thực sự tồn tại trong đúng module, không chỉ là pseudocode.
2. Các unit tests/smoke tests liên quan chạy thành công, có lệnh và kết quả thực tế.
3. README/config cập nhật; không có đường dẫn máy cá nhân hoặc secrets hardcode.
4. Có ví dụ chạy nhỏ trên CPU; phần cần GPU có hướng dẫn Kaggle.
5. Mọi benchmark nêu rõ môi trường và không tự bịa số liệu.
6. Codex tóm tắt: **Đã làm / File thay đổi / Test đã chạy / Kết quả / Vấn đề còn lại / Bước tiếp theo** và dừng chờ xác nhận.

## 7. Prompt đầu tiên dành cho Codex (sao chép nguyên văn)

> Hãy đọc kỹ `IMPLEMENTATION_PLAN.md` và kiểm tra cấu trúc repository hiện tại. Trước mắt **chỉ triển khai M0 và M1**, không thực hiện M2 trở đi. Tạo cấu trúc dự án, cấu hình môi trường, tải/chuẩn bị TinyStories cho Kaggle và **tự lập trình byte-level BPE Tokenizer từ đầu bằng Python** với vocabulary mục tiêu 8.000 token. Viết unit tests cho encode/decode, Unicode, save/load, determinism, merge rules và Dataset causal LM. Tạo `notebooks/01_tokenizer.ipynb` chỉ làm nhiệm vụ điều phối và import code từ `src`. Ưu tiên test trên tập nhỏ trước, tối ưu hiệu năng khi cần; không dùng tokenizer pretrained hay thuật toán BPE có sẵn. Chạy các test khả thi, báo cáo kết quả thực tế bằng tiếng Việt, liệt kê các file đã tạo và **dừng chờ tôi duyệt** trước khi chuyển sang Dense GPT.

---

**Thứ tự bắt đầu:** M0 → M1 → kiểm thử/duyệt → M2 → M3 → M4 → M5/M6 → M7. Khi thiếu thời gian, bảo vệ chất lượng và tính tái lập của M0–M4 trước khi mở rộng.
