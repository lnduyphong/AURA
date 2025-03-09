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
URL của model: http://localhost:9090
## 🎨 3. Mở Giao Diện Label Studio
Sau khi backend và mô hình đã được khởi động, mở giao diện Label Studio bằng lệnh:

```bash
label-studio
```
Truy cập giao diện tại:
➡️ http://localhost:8080

## 🔗 4. Kết Nối Label Studio Với Aura
Truy cập http://localhost:8080
Tạo project, vào phần setting để kết nối với model aura
![image](https://github.com/user-attachments/assets/6dcaae5e-81db-45a4-a235-8bdb3461098a)

![image](https://github.com/user-attachments/assets/7d1ae12b-4599-4899-9de0-f27eca60592e)

## 🔗 5. Gán nhãn
Lựa chọn các điểm dữ liệu mong muốn gán nhãn, click Retrivel Predictions
![image](https://github.com/user-attachments/assets/ca81086c-70b4-43c2-891d-317994dbd55c)


