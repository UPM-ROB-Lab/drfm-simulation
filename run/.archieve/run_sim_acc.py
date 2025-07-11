import yaml
import numpy as np
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import Simulator

# 加载配置文件
with open('config.yaml', encoding='utf-8') as f:  # 假设 config.yaml 在上一级目录
    config = yaml.safe_load(f)

# 计算理想圆形轨迹
steering_angle = 20  # 度
wheelbase = 0.6  # 车辆轴距，从config中获取

# 计算转弯半径 (阿克曼转向公式)
turning_radius = wheelbase / np.tan(np.radians(steering_angle))

# 生成圆形轨迹点
theta = np.linspace(0, 2*np.pi, 1000)  # 生成1000个点来形成圆
# 根据转向方向设置圆心位置
if steering_angle > 0:  # 左转
    circle_center_x = -turning_radius
    circle_center_y = 0
else:  # 右转
    circle_center_x = turning_radius
    circle_center_y = 0

# 生成目标轨迹点
target_trajectory = [(circle_center_x + turning_radius * np.sin(t),
                     circle_center_y - turning_radius * np.cos(t)) for t in theta]

# 创建仿真器实例，设置 plot_mode为'show'以显示轨迹图
sim = Simulator(config, dt=0.01, total_time=2, plot_mode='trac', target_trajectory=target_trajectory)

# 运行转向仿真
results = sim.run_steering_simulation(
    max_speed=6.0,
    accel_time=0,
    const_time=2,
    steering_angle=10,
    ackermann_ratio=1,
    speed_changed=True,
    angle_changed=True,
)

# 打印仿真结果
print(f"Simulation results:")
# print(f"  Average Slip Ratio: {results['avg_slip_ratio']:.4f}")
print(f"  Average Trajectory Error: {results['avg_trajectory_error']:.4f} m")
# print(f"  Average Steering Accuracy: {results['avg_steering_error']:.4f} degrees")
print(f"  Distance Travelled: {results['total_distance_travelled']:.4f} m")
print(f"  Total Energy Consumption: {results['total_energy_consumptions']:.4f} J")
print(f" Average Energy Consumption: {results['total_energy_consumptions']/results['total_distance_travelled']:.4f} J/m")
print(f"  Vehicle data saved to: {results['vehicle_data_path']}")
print(f"  Wheel data saved to: {results['wheel_data_path']}")
