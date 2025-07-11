import yaml
import numpy as np
from simulator import Simulator
import time
import datetime
from pathlib import Path

# 加载配置文件
with open('config.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

sim = Simulator(config, dt=0.01, total_time=10, plot_mode='save')

total_sim_time = 0
while total_sim_time < 10:  # 运行一段时间
    steering_angle = np.random.uniform(-30, 30)  # 随机生成转向角
    duration = 2  # 每个转向持续 2 秒
    num_steps = int(duration / sim.dt)

    speed_profile = np.full(num_steps, 5.0) # 假设速度恒定

    print(f"Applying steering angle: {steering_angle:.2f} degrees for {duration:.2f} seconds")

    for _ in range(num_steps):
        current_time_in_episode = _ * sim.dt
        relative_steering_angle = steering_angle # 可以根据时间变化调整转向角
        sim.explicit_euler(speed_profile[_], angles=[relative_steering_angle, relative_steering_angle]) # 简化，假设左右轮转向相同
        total_sim_time += sim.dt
        if total_sim_time >= 10:
            break
    if total_sim_time >= 10:
        break

# 创建输出目录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f'data/autonomous_output_{timestamp}')
output_dir.mkdir(parents=True, exist_ok=True)

# 保存结果
sim.plot_and_save_results(output_dir)
