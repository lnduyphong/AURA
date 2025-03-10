<div align="center">

# AURA: Autonomous Universal Refinement of Annotation
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Elsevier](https://img.shields.io/badge/📝-Paper-red)](https://www.sciencedirect.com/science/article/abs/pii/S0167739X2500024X#:~:text=In%20this%20paper%2C%20we%20introduce%20Cola%2C%20a%20novel,comprehensive%20and%20robust%20solution%20to%20corrupted%20label%20detection.)
[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/) 
</div>

# Giới thiệu
**AURA** là một giải pháp tiên tiến giúp **tự động hóa quá trình gán nhãn dữ liệu** với độ chính xác cao, đồng thời phát hiện và sửa lỗi nhãn sai. Hệ thống này kết hợp sức mạnh của **mô hình ngôn ngữ lớn (LLMs)** với **phân tích mối quan hệ dữ liệu** để tối ưu hóa chất lượng tập nhãn. Dự án được phát triển trong khuôn khổ **Hội nghị Sinh viên Nghiên cứu Khoa học cấp Khoa năm học 2024-2025** tại **Trường Đại học Công nghệ – ĐHQGHN**, nhằm cải thiện đáng kể **hiệu quả** và **độ chính xác** của quá trình gán nhãn dữ liệu trong các dự án **học máy (ML) và học sâu (DL)**.

# Cách cài đặt
### 1. Cài Đặt Thư Viện
Để sử dụng AURA, đầu tiên cần tải xuống các thư viên cần thiết với lệnh dưới đây:

```bash
cd backend
pip install -e .
```
### 2. Import OPENAI API
Với linux/macOS:
```bash
export OPENAI_API_KEY="your_api_key_here"
```
Với window:
```bash
setx OPENAI_API_KEY "your_api_key_here"
```

### 3. Khởi Chạy Mô Hình (Model)
Chạy mô hình bằng Docker Compose:

```bash
cd panda
docker compose up -d
```

URL của model: http://localhost:9090

### 4. Mở Giao Diện Label Studio
Sau khi backend và mô hình đã được khởi động, mở giao diện Label Studio bằng lệnh:

```bash
label-studio
```
Truy cập giao diện tại: http://localhost:8080

### 

### 4. Kết Nối Label Studio Với Aura
Truy cập http://localhost:8080
Tạo project
![image](https://github.com/user-attachments/assets/ffe87ebd-f53a-411a-a17b-0eea92aee79e)
Upload data cần gán nhãn
![image](https://github.com/user-attachments/assets/2e1e749d-587d-4f4e-a0fa-a69e73314c03)
Lựa chọn bài toàn
![image](https://github.com/user-attachments/assets/4ad28342-b600-4be1-87fc-397043cf7327)

Model AURA phục vụ bài toán text classification
Ở đây chúng tôi sample sẵn 2 bộ data Ag'news, Clickbait 
Có thể sử dụng file labels.txt để điền các nhãn dữ liệu, sau khi điền đầy đủ thông tin, click Save
![image](https://github.com/user-attachments/assets/8e29d57b-ef37-42b7-8703-61312d00517b)

Sau khi hoàn thành qúa trình chuẩn bị data, để kết nối với model: Chọn project -> setting -> Model
![image](https://github.com/user-attachments/assets/6dcaae5e-81db-45a4-a235-8bdb3461098a)

![image](https://github.com/user-attachments/assets/7d1ae12b-4599-4899-9de0-f27eca60592e)

# Sử dụng AURA
Khi đã kết nối thành công, lựa chọn các điểm dữ liệu mong muốn gán nhãn, click Retrivel Predictions
![image](https://github.com/user-attachments/assets/ca81086c-70b4-43c2-891d-317994dbd55c)


