# GWO - Tổng hợp bài tập/đồ án

Repo này gồm 2 phần độc lập:

1) **CNN-Handwritten_Digit**: Nhận diện chữ số viết tay bằng CNN, có phiên bản cơ bản và phiên bản tối ưu hyperparameters bằng **Grey Wolf Optimization (GWO)**.
2) **Digital_Signal_Processing-JCAS_Multibeam**: Tối ưu đa chùm tia (multibeam) cho **Joint Communication and Sensing (JCAS)** trong MATLAB.

## Cấu trúc thư mục

- [CNN-Handwritten_Digit](CNN-Handwritten_Digit)
  - `cnn_basic.py`: CNN cơ bản (hyperparameters khởi tạo ngẫu nhiên hoặc chỉ định)
  - `cnn_gwo.py`: CNN + GWO tối ưu `learning_rate` và `num_filters`
  - `compare_methods.py`: Chạy và so sánh 2 phương pháp
  - `requirements.txt`: Phụ thuộc Python

- [Digital_Signal_Processing-JCAS_Multibeam](Digital_Signal_Processing-JCAS_Multibeam)
  - `main.m`: Ví dụ chạy tạo beam/multibeam
  - `main_comparison.m`: So sánh các thuật toán (ILS, GWO, IGWO, ChaoticGWO, ...)
  - Các hàm hỗ trợ: `twoStepILS.m`, `GWO.m`, `IGWO.m`, `ChaoticGWO.m`, ...

## 1) CNN-Handwritten_Digit (Python)

### Yêu cầu

- Python 3.x
- Cài dependencies:

  - `pip install -r CNN-Handwritten_Digit/requirements.txt`

### Dữ liệu

Code đang đọc dữ liệu từ thư mục `archive` theo cấu trúc sau (mỗi class là 1 thư mục):

```
CNN-Handwritten_Digit/
  archive/
    0/  (các ảnh .jpg)
    1/
    ...
    9/
```

Mặc định các ảnh sẽ được đọc dạng grayscale và resize về `28x28`.

### Chạy so sánh (khuyến nghị)

Chạy script so sánh 2 phương pháp (CNN random hyperparams vs CNN+GWO):

```
python CNN-Handwritten_Digit/compare_methods.py
```

Kết quả sẽ tạo các file output như `comparison_results.json`, các hình `results_*.png`, và model `.h5` (tùy theo script).

## 2) Digital_Signal_Processing-JCAS_Multibeam (MATLAB)

### Chạy nhanh

Mở MATLAB, chuyển working directory sang thư mục:

- [Digital_Signal_Processing-JCAS_Multibeam](Digital_Signal_Processing-JCAS_Multibeam)

Sau đó chạy một trong các file:

- `main.m`: tạo beam/multibeam minh họa
- `main_comparison.m`: chạy và so sánh nhiều thuật toán tối ưu (ILS/GWO/IGWO/ChaoticGWO, ...)

### Ghi chú

Trong thư mục này có README tiếng Anh với mô tả ngắn và liên kết tham khảo:

- [Digital_Signal_Processing-JCAS_Multibeam/README.md](Digital_Signal_Processing-JCAS_Multibeam/README.md)

## Ghi chú repo

- File báo cáo LaTeX `bao-cao.tex` được **loại khỏi Git** (không upload) theo yêu cầu.
