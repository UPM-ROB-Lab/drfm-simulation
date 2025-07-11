import yaml
import numpy as np
from simulator import Simulator
import datetime
from pathlib import Path

# 加载配置文件
with open('config.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 创建父文件夹
parent_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
parent_dir = Path(f'batch_output/batch_output_{parent_timestamp}')
parent_dir.mkdir(parents=True, exist_ok=True)

ackermann_ratios = [0.6, 1.0, 1.4]
steering_angle = 20  # 可以从config.yaml获取或作为参数传递

for ratio in ackermann_ratios:
    print(f"\nRunning simulation with Ackermann ratio: {ratio}")
    # 创建子文件夹
    iteration_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = parent_dir / f"{steering_angle}deg_{ratio}_{iteration_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = Simulator(config, dt=0.01, total_time=10, plot_mode='save')
    results = sim.run_steering_simulation(
        max_speed=3.0,
        accel_time=1,
        const_time=8,
        steering_angle=steering_angle,
        ackermann_ratio=ratio,
        speed_changed=True,
        angle_changed=True,
    )
    # 保存结果到子文件夹
    results = sim.plot_and_save_results(output_dir)
    print(f"  Vehicle data saved to: {results['vehicle_data_path']}")
    print(f"  Wheel data saved to:   {results['wheel_data_path']}")

    # TODO: 在这里计算并记录滑转率和做功情况
    # 滑转率的计算可能需要在 Vehicle 或 Wheel 类中实现，并记录下来
    # 做功情况可以从保存的 CSV 文件中读取并分析（例如，计算总做功或最小瞬时做功）
    import pandas as pd
    wheel_data = pd.read_csv(results['wheel_data_path'])
    for i in range(1, 5):
        work_col = f'wheel_{i}_work'
        if work_col in wheel_data.columns:
            min_work = wheel_data[work_col].min()
            print(f"  Wheel {i} minimum instantaneous work: {min_work:.4f}")

    # 为了演示，我们简单地打印出 CSV 文件的路径
    print(f"  Raw data saved for Ackermann ratio {ratio}.")
