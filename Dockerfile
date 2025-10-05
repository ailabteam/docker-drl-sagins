# ==================================================================
# Giai đoạn 1: BUILDER - Cài đặt môi trường và đóng gói
# ==================================================================
# Sử dụng base image có đầy đủ công cụ build và CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Cài các gói hệ thống cần thiết cho việc cài đặt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Cài công cụ conda-pack để nén môi trường
RUN conda install -c conda-forge conda-pack

# Sao chép file định nghĩa môi trường vào image
COPY environment.yml .

# Tạo môi trường conda từ file
RUN conda env create -f environment.yml

# Nén môi trường đã tạo thành file .tar.gz
RUN conda-pack -n drl_env -o /tmp/env.tar.gz

# ==================================================================
# Giai đoạn 2: FINAL - Image cuối cùng, gọn nhẹ
# ==================================================================
# Bắt đầu từ một base image nhẹ hơn, chỉ chứa runtime CUDA
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Cài các gói hệ thống tối thiểu cần thiết để chạy code (ví dụ: cho OpenCV, Matplotlib)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxext6 \
    libsm6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục chứa môi trường
ENV CONDA_ENV_DIR /opt/conda-env
RUN mkdir -p $CONDA_ENV_DIR

# Sao chép file môi trường đã nén từ giai đoạn "builder"
COPY --from=builder /tmp/env.tar.gz .

# Giải nén môi trường và xóa file nén
RUN tar -xzf env.tar.gz -C $CONDA_ENV_DIR && \
    rm env.tar.gz

# ==================================================================
# === KHỐI LỆNH ĐÃ ĐƯỢC SỬA LỖI (LẦN 2) ===
# BƯỚC 1: Thiết lập PATH trước để hệ thống biết nơi tìm 'python'
ENV PATH=$CONDA_ENV_DIR/bin:$PATH

# BƯỚC 2: Bây giờ mới chạy conda-unpack
RUN conda-unpack
# ==================================================================

# Thiết lập thư mục làm việc mặc định
WORKDIR /workspace

# Lệnh mặc định khi container khởi động là mở terminal
CMD ["/bin/bash"]
