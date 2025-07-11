import numpy as np
from rft import RFTCalculator
import yaml
import math

class Wheel:
    """
    Represents a single wheel, including its position, speed, and interactions with the terrain.
    """
    def __init__(self, position=(0, 0, 0),offset=(0,0,0), phi=(0.0, 0.0, 0.0)):
        """
        Initialize a wheel with its position and geometry.
        :param position: Tuple (x, y, z) of the wheel's position in the vehicle's coordinate frame.
        :param radius: The radius of the wheel.
        :param width: The width of the wheel.
        """
        self.axle_pos  = tuple(position)           # 轴心 (wx, wy, wz)
        self.offset    = np.array(offset, float)   # 偏心向量 ex, ey, ez  (车体系)
        self.angle = 0  # 轮子与车体坐标系的夹角 (提前初始化)

        # 用 position 属性专门存“轮心在车体系的位置”
        self.position = self._calc_center_pos()    # 初始化一次

        self.absolute_position = tuple(position)  # 轮子的绝对位置 (确保是 tuple)
        # self.radius = radius
        # self.width = width
        self.omega = [0.0, 0.0, 0.0]  # Angular velocity vector (rad/s) [wx, wy, wz] 车轮自身的转速
        self.phi = np.array(phi)    # Angular position (rad) [rx, ry, rz] 车轮此时应该转到哪里了
        self.velocity = [0.0, 0.0, 0.0]  # (vx, vy, vz) of wheel's center
        self.forces = [0.0 ,0.0 ,0.0]
        self.omega_history = []  # Track omega over time
        self.torque = [0.0, 0.0, 0.0] # Store torque as a vector
        self.element_power = 0.0 # 新增：存储基于面元计算的功率

        # Define minimum vertical velocity to ensure numerical stability
        self.MIN_VERTICAL_VELOCITY = -1e-6  # 很小的向下速度
        self.rolling_radius = 0.1  # 添加滚动半径初始化 (单位: 米)
        
        # Initialize RFTCalculator with required parameters
        # Load RFT parameters from config.yaml
        with open('config.yaml', 'r',encoding="utf-8") as f:
            config = yaml.safe_load(f)
        rft_config = config.get('rft', {})
        self.rft_params = {
            'left_shape_matrix_path': rft_config['left_shape_matrix_path'],
            'right_shape_matrix_path': rft_config['right_shape_matrix_path'],
            'coefficients_path': rft_config['coefficients_path']
        }
        RFTCalculator.get_instance(rft_config) # 传递完整的 rft 配置

    def _calc_center_pos(self):
        # 旋转偏心向量到车体系
        c, s = math.cos(self.angle), math.sin(self.angle)
        ex, ey, ez = self.offset
        dx = ex * c - ey * s
        dy = ex * s + ey * c
        ax, ay, az = self.axle_pos
        return (ax + dx, ay + dy, az + ez)
    
    def get_roll_speed(self):
        """计算滚动速度 v_roll(t) = r * omega(t)"""
        # 假设角速度 omega 是绕 x 轴的角速度，即 self.omega[0]
        # 注意角速度 omega 的正负号需要根据速度方向和旋转方向统一约定
        return self.rolling_radius * abs(self.omega[0]) # 取角速度绝对值，滚动速度为正值

    def set_omega(self, omega_value):
        """
        Set the angular velocity of the wheel.
        :param omega_value: New angular velocity (rad/s) around x-axis.
        """
        
        # Convert scalar to vector with rotation around x-axis
        self.omega = [float(-omega_value), 0.0, 0.0]
        
        self.omega_history.append(self.omega)
        
    def set_angle(self, angle_value):
        self.angle = angle_value
        self.position = self._calc_center_pos()        # 刷新轮心坐标

    def get_force(self):
        return self.forces
    
    def get_torque(self):
        return self.torque
    
    def get_position(self):
        return self.position
    
    def get_phi(self):
        return self.phi

    def update_phi(self, dt):
        """
        Update the angular position of the wheel based on its angular velocity.
        :param dt: Time step duration.
        """
        self.phi += np.array(self.omega) * dt
        
    def update_velocity(self, velocity):
        """
        Update the linear velocity of the wheel.
        :param velocity: Tuple (vx, vy, vz) of the wheel's linear velocity.
        """
        self.velocity = velocity
        
    def update_absolute_position(self, veh_pos):
        """
        veh_pos = (x, y, z, theta)  —— 车辆质心在惯性系的位置与偏航角
        """
        x, y, z, theta = veh_pos
        wx, wy, wz = self.position          # 轮心在车体系
        cx = x + wx
        cy = y + wy
        cz = z + wz
        self.absolute_position = (cx, cy, cz)
        
    def calculate_forces(self, dt):
        """
        Calculate the three-axis forces acting on the wheel using RFT.
        :return: Tuple of forces (Fx, Fy, Fz) in vehicle's coordinate frame
        """
        # Get singleton RFTCalculator instance
        rft_calculator = RFTCalculator.get_instance()
        
        # Convert velocity to numpy array for calculations
        velocity = np.array(self.velocity)
        
        # Check if wheel is effectively static (both linear and angular velocity near zero)
        is_static = (np.all(np.abs(velocity) < 1e-10) and 
                    np.all(np.abs(self.omega) < 1e-10))
        
        # If wheel is static, add minimum vertical velocity
        if is_static:
            velocity[2] = self.MIN_VERTICAL_VELOCITY
        
        # Prepare wheel data
        wheel_data = {
            'absolute_position': self.absolute_position,
            'position': self.position,
            'velocity': velocity,
            'omega': self.omega,
            'phi': self.phi
        }
        
        # Calculate forces, torque, and power using RFT
        forces, torque, total_element_power = rft_calculator.compute_forces(wheel_data, dt)

        # Update wheel forces, torque, power, and phi
        self.forces = forces
        self.torque = torque # Store the torque vector
        self.element_power = total_element_power # 存储计算得到的功率
        self.torque = torque
        self.update_phi(dt)
        
        # Return forces as a 3-element tuple (Fx, Fy, Fz)
        return tuple(forces)
