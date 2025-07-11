import yaml
import numpy as np
from simulator import Simulator
import datetime
from pathlib import Path
import pandas as pd

# 加载配置文件
with open('config.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 创建父文件夹
parent_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
parent_dir = Path(f'batch_output/ackermann_comparison_{parent_timestamp}')
parent_dir.mkdir(parents=True, exist_ok=True)

# 定义测试条件
steering_angle = 14  # 转向角度
ackermann_ratios = [0.8, 1.0, 1.2]  # 阿克曼比率
angle_changed_options = [True, False]  # 是否启用阿克曼转向
speed_changed_options = [True, False]  # 是否启用速度变化

for angle_changed in angle_changed_options:
    for speed_changed in speed_changed_options:
        for ratio in ackermann_ratios:
            print(f"\nRunning simulation with Ackermann ratio: {ratio}, angle_changed: {angle_changed}, speed_changed: {speed_changed}")
            
            # 创建子文件夹
            iteration_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = parent_dir / f"{'with' if angle_changed else 'without'}_ackermann_{'with' if speed_changed else 'without'}_speed_{steering_angle}deg_{ratio}_{iteration_timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 运行仿真
            sim = Simulator(config, dt=0.01, total_time=14, plot_mode='save')
            results = sim.run_steering_simulation(
                max_speed=6.0,
                accel_time=2,
                const_time=10,
                steering_angle=steering_angle,
                ackermann_ratio=ratio,
                speed_changed=speed_changed,
                angle_changed=angle_changed,
            )
            
            # 保存结果
            results = sim.plot_and_save_results(output_dir)
            print(f"  Vehicle data saved to: {results['vehicle_data_path']}")
            print(f"  Wheel data saved to:   {results['wheel_data_path']}")

            # 分析轮子做功情况
            wheel_data = pd.read_csv(results['wheel_data_path'])
            for i in range(1, 5):
                work_col = f'wheel_{i}_work'
                if work_col in wheel_data.columns:
                    min_work = wheel_data[work_col].min()
                    print(f"  Wheel {i} minimum instantaneous work: {min_work:.4f}")

            print(f"  Raw data saved for Ackermann ratio {ratio}, angle_changed {angle_changed}, speed_changed {speed_changed}.")
