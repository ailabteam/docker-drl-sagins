import os
import datetime
import numpy as np
import matplotlib
# Dòng này rất quan trọng để matplotlib chạy được trong Docker
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 1. Thiết lập Thư mục Kết quả ---
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join("results", timestamp)
figures_dir = os.path.join(results_dir, "figures")
logs_dir = os.path.join(results_dir, "logs")

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

print(f"INFO: Results will be saved in: {results_dir}")

# --- 2. Mô phỏng quá trình huấn luyện ---
total_timesteps = 500
all_rewards = []

print("INFO: Starting simulated training...")
for step in range(total_timesteps):
    reward = 1 - np.exp(-step / 100) + np.random.rand() * 0.1
    all_rewards.append(reward)
    if (step + 1) % 100 == 0:
        print(f"  - Step [{step + 1}/{total_timesteps}], Reward: {reward:.4f}")
print("INFO: Training finished.")

# --- 3. Lưu dữ liệu thô ---
log_filepath = os.path.join(logs_dir, "rewards_log.csv")
np.savetxt(log_filepath, np.array(all_rewards), delimiter=",", header="reward", comments="")
print(f"INFO: Training log saved to: {log_filepath}")

# --- 4. Vẽ và Lưu Biểu đồ ---
plt.figure(figsize=(10, 5))
plt.plot(all_rewards, color='blue')
plt.title('Simulated Reward Over Time')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.grid(True)

reward_fig_path = os.path.join(figures_dir, "reward_plot.png")
plt.savefig(reward_fig_path)
plt.close()
print(f"INFO: Reward plot saved to: {reward_fig_path}")

print("\nSUCCESS: Experiment finished. Check the results directory.")
