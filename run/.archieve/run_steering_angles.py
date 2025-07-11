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
parent_dir = Path(f'batch_output/steering_angles_{parent_timestamp}')
parent_dir.mkdir(parents=True, exist_ok=True)

# 定义测试条件
steering_angles = [10]  # 不同转向角度
ackermann_ratio = 1.0  # 100%阿克曼转向

for angle in steering_angles:
    print(f"\nRunning simulation with steering angle: {angle} degrees")
    
    # 创建子文件夹
    iteration_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = parent_dir / f"{angle}deg_{ackermann_ratio}_{iteration_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行仿真
    sim = Simulator(config, dt=0.01, total_time=6, plot_mode='save')
    results = sim.run_steering_simulation(
        max_speed=3.0,
        accel_time=2,
        const_time=2,
        steering_angle=angle,
        ackermann_ratio=ackermann_ratio,
        speed_changed=True,
        angle_changed=True,
    )
    
    # 保存结果
    results = sim.plot_and_save_results(output_dir)
    print(f"  Vehicle data saved to: {results['vehicle_data_path']}")
    print(f"  Wheel data saved to:   {results['wheel_data_path']}")

    # 分析轮子做功情况
    import pandas as pd
    wheel_data = pd.read_csv(results['wheel_data_path'])
    for i in range(1, 5):
        work_col = f'wheel_{i}_work'
        if work_col in wheel_data.columns:
            min_work = wheel_data[work_col].min()
            print(f"  Wheel {i} minimum instantaneous work: {min_work:.4f}")

    print(f"  Raw data saved for steering angle {angle} degrees.")
