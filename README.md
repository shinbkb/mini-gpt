# MiniGPT-Sys

Dự án xây dựng Decoder-Only Transformer từ đầu bằng PyTorch.
Hiện chỉ triển khai **M0 — khởi tạo repository và môi trường**.
Kế hoạch: [IMPLEMENTATION_PLAN_VI.md](IMPLEMENTATION_PLAN_VI.md).

## Cấu trúc

- `configs/`: cấu hình mục tiêu Dense và MoE.
- `src/`: package, module kiểm tra môi trường và các file khung theo kế hoạch.
- `tests/`: smoke test CPU; test tokenizer/dataset còn là file khung.
- `notebooks/00_setup.ipynb`: mẫu thiết lập Kaggle, khóa commit nguồn.
- `notebooks/01_tokenizer.ipynb` đến `main_pipeline.ipynb`: notebook khung.
- `experiments/`, `reports/`, `artifacts/`: thí nghiệm, báo cáo, output.

Tokenizer, Dataset, model, train/evaluate và benchmark chưa triển khai.
Hai YAML là cấu hình mục tiêu cho milestone sau, chưa được dùng để train.

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

## Phạm vi nghiệm thu M0

Import `src`, chạy smoke test CPU và báo cáo môi trường. Không tải TinyStories,
không huấn luyện tokenizer/model. Chờ duyệt M0 trước khi triển khai M1.
