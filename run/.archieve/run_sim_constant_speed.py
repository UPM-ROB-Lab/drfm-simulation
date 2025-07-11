import yaml
import numpy as np
from simulator import Simulator

"""
匀速转向仿真入口脚本
- 设定转向幅度为20度
- 轮子角速度为5rad/s
- 全程匀速运动，速度为5m/s
"""

# 加载配置文件
with open('config.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 创建仿真器实例
sim = Simulator(config, dt=0.01, total_time=10)

# 运行转向仿真
# - max_speed: 匀速5m/s
# - accel_time: 加速时间0s (无加速过程)
# - const_time: 匀速时间10s (全程匀速)
# - steering_angle: 转向角度20度
# - wheel_angular_velocity: 轮子角速度5rad/s
results = sim.run_steering_simulation(
    max_speed=12.0,
    accel_time=0,
    const_time=10,
    steering_angle=25,
    ackermann_ratio=1.2,
    angle_changed=True, # 转向角度设置速度变化
    speed_changed=True # 阿克曼设置速度差速
)
