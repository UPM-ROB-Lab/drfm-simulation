from wheel import Wheel
from rft import RFTCalculator
import yaml
import math


class Vehicle:
    """
    Represents the entire vehicle, including its wheels and dynamic behavior.
    """

    def __init__(self, config):
        """
        Initialize the vehicle with configuration from config.yaml.
        :param config: Dictionary containing vehicle configuration
        """
        self.mass = config['vehicle']['mass']
        self.inertia_z = config['vehicle']['inertia_z']
        self.rft_calculator = RFTCalculator(config['rft'])
        self.position = config['vehicle']['initial_position']  # 从配置读取初始位置 [x, y, z, theta]
        self.linear_damping_coefficient = config['vehicle'].get('linear_damping_coefficient', 0.0)  # 线性阻尼系数
        self.angular_damping_coefficient = config['vehicle'].get('angular_damping_coefficient', 0.0) # 角阻尼系数
        self.velocity = [0, 0, 0, 0]  # 初始速度 [vx, vy, vz, omega]
        self.wheels = self._load_wheels_from_config(config)  # 创建四个轮子
        # 初始化时更新轮胎绝对位置
        for wheel in self.wheels:
            wx, wy, wz = wheel.position
            x, y, z, theta = self.position
            wheel_x = x + wx * math.cos(theta) - wy * math.sin(theta)
            wheel_y = y + wy * math.cos(theta) + wx * math.sin(theta)
            wheel_z = z + wz
            wheel.absolute_position = (wheel_x, wheel_y, wheel_z)
        self.total_forces = [0, 0, 0, 0]  # [Fx, Fy, Fz, Tz]
        self.acceleration = [0, 0, 0, 0]  # [ax, ay, az, alpha]
        
        

    def _load_wheels_from_config(self, config):
        """Creates Wheel objects from configuration."""
        wheels = []
        wheel_configs = config.get("wheels", [])
        for wheel_config in wheel_configs:
            position = tuple(wheel_config.get("position", [0, 0, 0]))
            offset = tuple(wheel_config.get("offset", [0, 0, 0]))  # 从配置读取 offset
            wheel = Wheel(position=position, offset=offset) # 将 offset 传递给 Wheel 构造函数
            wheels.append(wheel)


        return wheels

    def input_speed_and_angle(self, speeds, angles):
        """_summary_
        每个时间步运行前，通过调用这个函数修改每个轮子的速度以及前向轮的两个角度用于动力学系统的输入
        之后调用RFTCalculator的compute_forces函数计算每个轮子的受力
        Args:
            speeds (拥有四个元素的数组(角速度)): 左前、右前、左后、右后四个轮子的速度
            angles (拥有两个元素的数组(弧度)): 左前、右前两个轮子的旋转角度
        """
        # 设置每个轮子的速度值
        for wheel, speed in zip(self.wheels, speeds):
            wheel.set_omega(speed)
        # 设置前两个轮子的转向角度
        self.wheels[0].set_angle(angles[0])
        self.wheels[1].set_angle(angles[1])
        
        return 0
        
    
    def update_forces(self, dt):
        """
        计算并更新车体整体所受到的三个方向上的合力和Z轴力矩
        返回：包含三个方向力和Z轴力矩的四元组 [Fx, Fy, Fz, Tz]
        """
        # 初始化合力和力矩为0
        self.total_forces = [0, 0, 0, 0]  # [Fx, Fy, Fz, Tz]

        # 获取车辆的转角
        theta = self.position[3]

        # 遍历每个轮子，计算合力和力矩
        for wheel in self.wheels:
            # 获取轮子的受力和位置
            wheel.calculate_forces(dt)
            wheel_forces = wheel.get_force()
            wheel_pos = wheel.get_position()

            # 获取轮子的转向角（如果存在）
            steering_angle = -wheel.angle  # 假设 Wheel 类有 angle 属性，由于我是回正，因此我需要取负

            # 将轮子在自身坐标系下的力旋转到车体坐标系
            fx_wheel, fy_wheel, fz_wheel = wheel_forces
            fx_body = fx_wheel * math.cos(steering_angle) - fy_wheel * math.sin(steering_angle)
            fy_body = fx_wheel * math.sin(steering_angle) + fy_wheel * math.cos(steering_angle)
            fz_body = fz_wheel # Z轴方向通常不受转向影响

            # 将轮子在车体坐标系下的力旋转到惯性坐标系
            fx_inertial = fx_body * math.cos(theta) - fy_body * math.sin(theta)
            fy_inertial = fy_body * math.cos(theta) + fx_body * math.sin(theta)

            # 累加三个方向的力 (使用旋转后的力)
            self.total_forces[0] += fx_inertial  # X方向 惯性坐标系
            self.total_forces[1] += fy_inertial  # Y方向 惯性坐标系
            self.total_forces[2] += fz_body      # Z方向 (Z轴通常与旋转无关，假设轮子力的Z方向也在车体坐标系下)

            # 计算并累加Z轴力矩 (第四分量) - 力矩的计算已经考虑了位置关系，不需要额外旋转
            torque = wheel_pos[0] * fy_body - wheel_pos[1] * fx_body # 注意这里使用车体坐标系下的力
            self.total_forces[3] += torque

        vx, vy, vz, _ = self.velocity
        linear_damping_fx = -self.linear_damping_coefficient * vx
        linear_damping_fy = -self.linear_damping_coefficient * vy
        linear_damping_fz = -self.linear_damping_coefficient * vz

        # self.total_forces[0] += linear_damping_fx
        # self.total_forces[1] += linear_damping_fy
        self.total_forces[2] += linear_damping_fz

        # 计算并应用角阻尼力矩
        omega = self.velocity[3]
        angular_damping_tz = -self.angular_damping_coefficient * omega

        self.total_forces[3] += angular_damping_tz

        return self.total_forces
    
    
    
    def update_acceleration(self):
        """
        根据输入的力与力矩计算车体整体的加速度与角加速度
        """
        # 从total_forces获取合力和力矩 惯性坐标系
        fx = self.total_forces[0]
        fy = self.total_forces[1] 
        fz = self.total_forces[2]
        tz = self.total_forces[3]
        
        # 计算线加速度 (F = ma)
        ax = fx / self.mass
        ay = fy / self.mass
        az = fz / self.mass - 9.81  # 减去重力加速度
        
        # 计算角加速度 (T = Iα) 逆时针为正
        alpha = tz / self.inertia_z
        
        # 更新加速度
        self.acceleration = [ax, ay, az, alpha]
        
        return self.acceleration
    
    

    def get_state(self):
        """
        Get the current state of the vehicle.
        :return: Dictionary of state variables.
        """
        return {
            "position": tuple(self.position),
            "velocity": tuple(self.velocity),
            "total_forces": tuple(self.total_forces)
        }

    def update_wheel_velocities(self):
        """
        Update the velocities of all wheels based on vehicle velocity and wheel positions.
        """
        vx_inertial, vy_inertial, vz_inertial, omega_z_body = self.velocity  # 惯性坐标系下的车辆速度
        theta = self.position[3]  # 获取车辆朝向角

        # 先将惯性坐标系下的速度转换到车体坐标系
        vx_body = vx_inertial * math.cos(theta) + vy_inertial * math.sin(theta)
        vy_body = -vx_inertial * math.sin(theta) + vy_inertial * math.cos(theta)
        vz_body = vz_inertial

        for wheel in self.wheels:
            # Get wheel position relative to vehicle center (车体坐标系)
            wx_body, wy_body, wz_body = wheel.position

            # 1. 计算轮心在车体坐标系下的速度
            v_wheel_x_body = vx_body - omega_z_body * wy_body
            v_wheel_y_body = vy_body + omega_z_body * wx_body
            v_wheel_z_body = vz_body

            # 2. 应用转向角旋转，将速度从车体坐标系转换到轮子坐标系
            v_wheel_x_wheel = v_wheel_x_body
            v_wheel_y_wheel = v_wheel_y_body
            v_wheel_z_wheel = v_wheel_z_body

            if hasattr(wheel, 'angle') and wheel.angle != 0:
                angle = wheel.angle
                v_wheel_x_wheel_rotated = v_wheel_x_body * math.cos(angle) - v_wheel_y_body * math.sin(angle)
                v_wheel_y_wheel_rotated = v_wheel_x_body * math.sin(angle) + v_wheel_y_body * math.cos(angle)
                v_wheel_x_wheel, v_wheel_y_wheel = v_wheel_x_wheel_rotated, v_wheel_y_wheel_rotated

            # Update wheel velocity (假设 wheel.update_velocity 期望轮子坐标系下的速度)
            wheel.update_velocity((v_wheel_x_wheel, v_wheel_y_wheel, v_wheel_z_wheel))
            
    def update_wheel_position(self):
        """
        Update the absolute positions of all wheels based on vehicle position and wheel positions.
        """
        x, y, z, theta = self.position
        
        for wheel in self.wheels:
            # 获取轮子在车体坐标系下的相对位置
            wx, wy, wz = wheel.position
            
            # 计算轮子的绝对位置
            wheel_x = x + wx * math.cos(theta) - wy * math.sin(theta)
            wheel_y = y + wy * math.cos(theta) + wx * math.sin(theta)
            wheel_z = z + wz
            
            # 更新轮子的绝对位置
            wheel.absolute_position = (wheel_x, wheel_y, wheel_z)
