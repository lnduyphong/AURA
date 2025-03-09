# 🏷️ Hướng Dẫn Cài Đặt AURA tích hợp với Label-Studio

## 📌 Giới Thiệu
**Label Studio** là một công cụ mạnh mẽ để gán nhãn dữ liệu cho Machine Learning.  
Hướng dẫn này sẽ giúp bạn cài đặt, chạy Label Studio, kết nối với mô hình và sử dụng để gán nhãn dữ liệu.

---

## ⚙️ 1. Cài Đặt Backend
Trước tiên, cài đặt Label Studio backend:

```bash
cd backend
pip install -e .
```
## 🚀 2. Khởi Chạy Mô Hình (Model)
Chạy mô hình bằng Docker Compose:

```bash
cd panda
docker compose up -d
```
## 🎨 3. Mở Giao Diện Label Studio
Sau khi backend và mô hình đã được khởi động, mở giao diện Label Studio bằng lệnh:

```bash
label-studio
```
Truy cập giao diện tại:
➡️ http://localhost:8080

## 🔗 4. Kết Nối Label Studio Với Aura
Truy cập http://localhost:8080

![image](https://github.com/user-attachments/assets/d7dc4033-c3a4-43b8-9fcb-abcd26ca9665)
