### **How to Deploy**

1.  **Open a terminal and navigate to your project directory:**
    ```bash
    cd ~/hao/git/working/docker-drl-sagins
    ```
2.  **Create or open the `README.md` file:**
    ```bash
    nano README.md
    ```
3.  **Copy the entire content below and paste it into the open `README.md` file.**
4.  **Save and close the file:** Press `Ctrl+X`, then `Y`, then `Enter`.
5.  **Commit and push the file to GitHub:**
    ```bash
    git add README.md
    git commit -m "docs: Create detailed English README"
    git push
    ```

---

### **Detailed Content for `README.md` (English Version)**

````markdown
# Reproducible Research Environment for DRL/SAGINs (GPU & Docker)

[![Docker Pulls](https://img.shields.io/docker/pulls/haodpsut/drl-sagin-env?style=flat-square)](https://hub.docker.com/r/haodpsut/drl-sagin-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

This is a meticulously crafted, containerized development environment designed for scientific research that demands high-performance computing (GPU-accelerated) and **absolute reproducibility**.

This environment is specifically tailored for research in fields such as:
*   Deep Reinforcement Learning (DRL)
*   Space-Air-Ground Integrated Networks (SAGINs)
*   Unmanned Aerial Vehicle (UAV) and Satellite Systems
*   Federated Learning (FL)

The primary goal is to definitively solve the "It works on my machine!" problem and provide a stable foundation for anyone to reproduce the experimental results presented in scientific papers.

---

## 🚀 What's Inside? (Core Components)

This Docker image is built on a solid foundation and includes a curated set of popular libraries, tested for compatibility.

| Category              | Technology / Library                                 | Version / Notes                                    |
| --------------------- | ---------------------------------------------------- | -------------------------------------------------- |
| **Base OS**           | Ubuntu 22.04                                         | A stable and widely-used operating system.         |
| **GPU Support**       | CUDA 12.1.1                                          | Broadly compatible with modern NVIDIA GPUs.        |
| **Environment Mgmt**  | Conda (via Miniconda)                                | Manages Python libraries in an isolated, robust way. |
| **Language**          | Python                                               | `3.10`                                             |
| **AI/ML Framework**   | PyTorch, TorchVision, TorchAudio                     | Built with `pytorch-cuda=12.1` for GPU support.    |
| **DRL Framework**     | Gymnasium, Stable-Baselines3                         | The standard toolkits for modern DRL research.     |
| **Compute & Data**    | NumPy, Pandas, SciPy, Scikit-learn                   | The core libraries of the data science stack.      |
| **Visualization**     | Matplotlib, Seaborn                                  | For plotting and saving experimental results.      |
| **Image Processing**  | OpenCV                                               | Useful for tasks involving UAV/satellite imagery.  |

---

## ⚙️ How Does It Work? (The Workflow)

The core idea is to decouple the code editing environment from the code execution environment.

1.  **You write code on your Host machine:** Use your favorite tools (VS Code, PyCharm, etc.) on your local machine or server to write and edit Python files.
2.  **The Docker Container is your "Lab":** This container holds all the complex library dependencies and is pre-configured to access the GPU.
3.  **Volume Mounting (`-v` flag) is the Bridge:** When you launch the container, you "mount" your project directory from the Host into the `/workspace` directory inside the container. Any change you make on the Host is instantly reflected inside the container, and vice-versa.
4.  **You run experiments inside the Container:** Execute `python` commands inside the container's terminal. Your code runs with the full power of the GPU and the pre-installed libraries.
5.  **Results appear on your Host machine:** When your code generates result files (e.g., `.png` plots, `.csv` logs) inside the `/workspace` directory, these files instantly appear in your project folder on the Host.

**This workflow gives you the best of both worlds: the convenience of coding on your familiar Host system and the power and consistency of a dedicated Docker environment.**

---

## ⚡ Quick Start Guide

Follow these steps to set up the environment and run a sample experiment.

### Prerequisites

Your machine must have the following installed:
1.  A recent [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx).
2.  [Docker Engine](https://docs.docker.com/engine/install/).
3.  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Step 1: Pull the Image from Docker Hub

Open a terminal and run the following command to download the Docker image:
```bash
docker pull haodpsut/drl-sagin-env:latest
```

### Step 2: Prepare Your Project Directory

If you haven't already, clone this repository (or create a new project directory with a similar structure).
```bash
git clone https://github.com/ailabteam/docker-drl-sagins.git
cd docker-drl-sagins
```

### Step 3: Launch the Container

From within your project directory (`docker-drl-sagins`), run the command below. This will start the container and mount your current directory into it.

```bash
docker run -it --rm \
  --gpus all \
  -v "$(pwd)":/workspace \
  haodpsut/drl-sagin-env:latest
```
*   `--gpus all`: **Required!** Grants the container access to all available GPUs.
*   `-v "$(pwd)":/workspace`: Mounts the current host directory into `/workspace` inside the container.

After running the command, your terminal prompt will change, indicating you are now inside the container.

### Step 4: Run a Sample Experiment

Inside the container, execute the sample `train.py` script to confirm that everything is working:
```bash
python src/train.py
```
You will see log messages printed to the screen, detailing the simulated training process and the file paths for the saved results.

### Step 5: Check the Results

1.  Once the script has finished, exit the container by typing:
    ```bash
    exit
    ```
2.  Now, back on your Host machine, inspect the `results` directory:
    ```bash
    ls -R results
    ```
    You will see a new directory named with a timestamp (e.g., `YYYY-MM-DD_HH-MM-SS`), which contains `reward_plot.png` and `rewards_log.csv`. You can open these files directly to view your results.

---

## 🔧 Customizing and Building from Source

If you need to add more libraries or change existing versions:

1.  Modify the `environment.yml` file to add or remove Conda packages.
2.  Rebuild the image with your own custom tag:
    ```bash
    docker build -t your-dockerhub-username/my-custom-env:latest .
    ```
