# Môi trường Nghiên cứu Tái tạo cho DRL/SAGINs (GPU & Docker)

[![Docker Pulls](https://img.shields.io/docker/pulls/haodpsut/drl-sagin-env)](https://hub.docker.com/r/haodpsut/drl-sagin-env)

Đây là một môi trường phát triển được đóng gói bằng Docker, xây dựng cẩn thận để phục vụ cho các nghiên cứu khoa học yêu cầu tính toán hiệu năng cao (GPU-accelerated) và **tính tái tạo tuyệt đối (full reproducibility)**.

Môi trường này được thiết kế đặc biệt cho các lĩnh vực:
*   Học tăng cường sâu (Deep Reinforcement Learning - DRL)
*   Mạng tích hợp Không gian-Mặt đất-Trên không (SAGINs)
*   Hệ thống máy bay không người lái (UAV) và Vệ tinh
*   Học tập Liên kết (Federated Learning)

Mục tiêu chính là giải quyết triệt để vấn đề "Nó chạy trên máy tôi!" ("It works on my machine!") và cung cấp một nền tảng vững chắc để bất kỳ ai cũng có thể tái tạo lại các kết quả thí nghiệm trong các bài báo khoa học.

---

## 🚀 Nó có gì? (Core Components)

Image Docker này được xây dựng trên một nền tảng vững chắc và bao gồm các thư viện phổ biến nhất, đã được kiểm tra tương thích với nhau.

| Thể loại              | Công nghệ / Thư viện                                 | Phiên bản / Ghi chú                                |
| --------------------- | ---------------------------------------------------- | -------------------------------------------------- |
| **Nền tảng**          | Ubuntu 22.04                                         | Hệ điều hành ổn định.                              |
| **GPU Support**       | CUDA 12.1.1                                          | Tương thích rộng rãi với các GPU NVIDIA hiện đại.    |
| **Quản lý môi trường** | Conda (thông qua Miniconda)                          | Quản lý thư viện Python một cách cô lập và hiệu quả. |
| **Ngôn ngữ**          | Python                                               | `3.10`                                             |
| **AI/ML Framework**   | PyTorch, TorchVision, TorchAudio                     | Phiên bản được build với `pytorch-cuda=12.1`.      |
| **DRL Framework**     | Gymnasium, Stable-Baselines3                         | Bộ công cụ tiêu chuẩn cho nghiên cứu DRL.           |
| **Tính toán & Dữ liệu** | NumPy, Pandas, SciPy, Scikit-learn                   | Các thư viện cốt lõi của khoa học dữ liệu.         |
| **Trực quan hóa**      | Matplotlib, Seaborn                                  | Vẽ và lưu biểu đồ kết quả thí nghiệm.              |
| **Xử lý Ảnh**         | OpenCV                                               | Hữu ích cho các bài toán xử lý ảnh từ UAV/vệ tinh. |

---

## ⚙️ Nó hoạt động thế nào? (The Workflow)

Ý tưởng cốt lõi là tách biệt môi trường viết code và môi trường thực thi code.

1.  **Bạn viết code trên máy Host:** Bạn sử dụng các công cụ yêu thích của mình (VS Code, PyCharm, etc.) trên máy tính cá nhân hoặc server để viết và chỉnh sửa các file Python.
2.  **Container Docker là "phòng thí nghiệm":** Container này chứa tất cả các thư viện phức tạp và được cấu hình để "nhìn thấy" GPU.
3.  **Cơ chế ánh xạ thư mục (`-v`):** Khi bạn khởi động container, bạn "ánh xạ" thư mục code của bạn trên máy Host vào thư mục `/workspace` bên trong container. Mọi thay đổi bạn thực hiện trên máy Host sẽ được phản ánh ngay lập tức bên trong container, và ngược lại.
4.  **Chạy thí nghiệm:** Bạn thực thi các lệnh `python` bên trong container. Code sẽ được chạy với toàn bộ sức mạnh của GPU và các thư viện đã được cài đặt sẵn.
5.  **Kết quả xuất hiện trên máy Host:** Khi code của bạn tạo ra các file kết quả (biểu đồ `.png`, log `.csv`) trong thư mục `/workspace`, các file này cũng sẽ ngay lập tức xuất hiện trong thư mục dự án trên máy Host của bạn.

**Quy trình này giúp bạn có được những lợi ích tốt nhất của cả hai thế giới: sự tiện lợi của việc code trên máy Host và sức mạnh, tính nhất quán của môi trường Docker.**

---

## ⚡ Bắt đầu Nhanh (Quick Start)

Làm theo các bước sau để thiết lập và chạy một thí nghiệm ví dụ.

### Yêu cầu

Máy của bạn phải được cài đặt:
1.  [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) phiên bản mới.
2.  [Docker Engine](https://docs.docker.com/engine/install/).
3.  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Bước 1: Kéo Image từ Docker Hub

Mở terminal và chạy lệnh sau để tải image về máy của bạn:
```bash
docker pull haodpsut/drl-sagin-env:latest
```

### Bước 2: Chuẩn bị Thư mục Dự án

Nếu bạn chưa có, hãy clone repository này (hoặc tạo một thư mục dự án mới với cấu trúc tương tự).
```bash
git clone https://github.com/ailabteam/docker-drl-sagins.git
cd docker-drl-sagins
```

### Bước 3: Khởi chạy Container

Từ bên trong thư mục dự án (`docker-drl-sagins`), chạy lệnh sau. Lệnh này sẽ khởi động container và ánh xạ thư mục hiện tại của bạn vào đó.

```bash
docker run -it --rm \
  --gpus all \
  -v "$(pwd)":/workspace \
  haodpsut/drl-sagin-env:latest
```
*   `--gpus all`: **Bắt buộc!** Cấp quyền cho container sử dụng GPU.
*   `-v "$(pwd)":/workspace`: Ánh xạ thư mục hiện tại vào `/workspace` trong container.

Sau khi chạy lệnh, dấu nhắc terminal của bạn sẽ thay đổi, cho biết bạn đang ở bên trong container.

### Bước 4: Chạy một Thí nghiệm Ví dụ

Bên trong container, hãy chạy script `train.py` mẫu để xác nhận mọi thứ hoạt động:
```bash
python src/train.py
```
Bạn sẽ thấy các dòng log in ra màn hình, thông báo về quá trình huấn luyện mô phỏng và các file kết quả được lưu.

### Bước 5: Kiểm tra Kết quả

1.  Sau khi script chạy xong, thoát khỏi container bằng lệnh:
    ```bash
    exit
    ```
2.  Bây giờ, trên máy Host của bạn, hãy kiểm tra thư mục `results`:
    ```bash
    ls -R results
    ```
    Bạn sẽ thấy một thư mục mới được tạo theo dạng `YYYY-MM-DD_HH-MM-SS`, bên trong chứa các file `reward_plot.png` và `rewards_log.csv`. Bạn có thể mở các file này để xem kết quả.

---

## 🔧 Tùy chỉnh và Build lại từ Nguồn

Nếu bạn muốn thêm thư viện hoặc thay đổi phiên bản:

1.  Chỉnh sửa file `environment.yml` để thêm hoặc xóa các gói `conda`.
2.  Build lại image với tag của riêng bạn:
    ```bash
    docker build -t your-dockerhub-username/my-custom-env:latest .
    ```
