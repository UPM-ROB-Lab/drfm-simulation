import numpy as np
import yaml
import matplotlib.pyplot as plt
import datetime
import json
import os
from simulator import Simulator

# --- 常量定义 ---
MIN_DISTANCE = 2.0            # 最小行驶距离 (m)
BASE_SPEED = 6.0              # 基准速度 (rad/s)
STEERING_ANGLE = 20           # 转向角 (degrees)
TOTAL_TIME = 6                # 总运行时间 (s)
ACCEL_TIME = 1                # 加速时间 (s)
CONST_TIME = 4                # 匀速时间 (s)
DT = 0.01                     # 仿真步长 (s)
PENALTY_MULTIPLIER = 1000     # 惩罚因子

if __name__ == "__main__":
    # --- 加载配置文件 ---
    try:
        with open('config.yaml', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        exit(1)
    
    # --- 生成目标轨迹 ---
    # 这里采用 theta 从 0 到 2π（生成完整圆）; 计算方法与原代码保持一致
    theta = np.linspace(0, 2 * np.pi, 250)
    wheelbase = config.get('vehicle', {}).get('wheelbase', 0.6)
    turning_radius = wheelbase / np.tan(np.radians(STEERING_ANGLE))
    if STEERING_ANGLE > 0:
        circle_center_x = -turning_radius
        circle_center_y = 0
    else:
        circle_center_x = turning_radius
        circle_center_y = 0
    target_trajectory = [(circle_center_x + turning_radius * np.sin(t),
                          circle_center_y - turning_radius * np.cos(t)) for t in theta]
    
    # --- Warm-up 单线程运行：第一次运行以编译 numba ---
    warmup_candidate = [0.5, 1.2, 0.75, 1.2]
    print("Starting warm-up simulation for Numba compilation...")
    sim_warmup = Simulator(config, dt=DT, total_time=TOTAL_TIME, plot_mode='none', target_trajectory=target_trajectory)
    _ = sim_warmup.run_steering_optimization_simulation(
        max_speed=BASE_SPEED,
        accel_time=ACCEL_TIME,
        const_time=CONST_TIME,
        steering_angle=STEERING_ANGLE,
        ackermann_ratio=warmup_candidate[0],
        beta_FR=warmup_candidate[1],
        beta_RL=warmup_candidate[2],
        beta_RR=warmup_candidate[3],
        speed_changed=True,
        angle_changed=True
    )
    print("Warm-up simulation completed.\n")
    
    # --- 手动输入最佳参数 ---
    print("请输入你希望使用的四个参数：")
    try:
        ackermann_ratio = 0.71
        beta_FR = 1.38
        beta_RL = 0.94
        beta_RR = 1.34
    except Exception as e:
        print("输入有误，使用默认参数 [0.6, 1.3, 0.8, 1.3]。")
        ackermann_ratio, beta_FR, beta_RL, beta_RR = 0.6, 1.3, 0.8, 1.3
        
    best_params = [ackermann_ratio, beta_FR, beta_RL, beta_RR]
    print(f"使用手动输入的参数：{best_params}")
    
    # --- 使用手动输入的参数进行最终仿真展示 ---
    sim_final = Simulator(config, dt=DT, total_time=TOTAL_TIME, plot_mode='trac', target_trajectory=target_trajectory)
    final_results = sim_final.run_steering_optimization_simulation(
        max_speed=BASE_SPEED,
        accel_time=ACCEL_TIME,
        const_time=CONST_TIME,
        steering_angle=STEERING_ANGLE,
        ackermann_ratio=best_params[0],
        beta_FR=best_params[1],
        beta_RL=best_params[2],
        beta_RR=best_params[3],
        speed_changed=True,
        angle_changed=True
    )
    
    print("\nFinal Simulation with Manually Provided Parameters:")
    print(f"Trajectory Error: {final_results['avg_trajectory_error']:.4f} m")
    print(f"Energy Consumption: {final_results['total_energy_consumptions']:.2f} J")
    print(f"Total Distance Travelled: {final_results['total_distance_travelled']:.2f} m")
    
    # 如果需要保存结果或轨迹图，可以在这里添加对应的保存逻辑
    # 例如：保存最终轨迹图
    output_folder = os.path.join("simulation_results", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_folder, exist_ok=True)
    trajectory = np.array(sim_final.positions)
    target_arr = np.array(target_trajectory)
    
    plt.figure(figsize=(8, 6))
    plt.plot(target_arr[:, 0], target_arr[:, 1], 'r--', linewidth=2, label='Ideal Trajectory')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Actual Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title("Final Simulation Trajectory")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_folder, "final_trajectory.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Final trajectory plot saved to {save_path}")
