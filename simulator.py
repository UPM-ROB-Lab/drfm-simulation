from tqdm import tqdm
# simulator.py

import numpy as np
import time # 添加 time 模块
from vehicle import Vehicle
from plotter import Plotter
from vehicle_metrics import VehicleMetrics
# 全局积分方法选择：'euler'（显式欧拉法）或 'rk4'（四阶龙格库塔法）
INTEGRATION_METHOD = 'euler'

class Simulator:
    def __init__(self, config, dt, total_time, plot_mode='show', target_trajectory=None, target_steering_angle=0.0, circle_center=None, turning_radius=None, progress_callback=None): # Add new param progress_callback
        """
        初始化仿真器
        :param config: 配置字典
        :param dt: 时间步长 (s)
        :param total_time: 总仿真时间 (s)
        :param plot_mode: 图片显示模式 ('show', 'save', 'none')
        :param target_trajectory: 目标轨迹点数组 (N, 2)
        :param target_steering_angle: 目标转向角 (用于指标计算)
        :param circle_center: 理论目标圆心坐标 (x, y), 默认为 None
        :param turning_radius: 理论目标转向半径, 默认为 None
        """
        # 使用 hasattr 检查确保只初始化一次或按需重新初始化
        # if not hasattr(self, 'initialized') or self.dt != dt or self.total_time != total_time:
        self.vehicle = Vehicle(config)
        self.dt = dt
        self.total_time = total_time
        self.time_steps = int(total_time / dt)
        self.time = np.linspace(0, total_time, self.time_steps)
        self.plot_mode = plot_mode  # 保存绘图模式
        self.target_trajectory = target_trajectory # Keep original trajectory for plotting etc.
        self.target_steering_angle = target_steering_angle # Keep for other potential uses
        self.circle_center = np.array(circle_center) if circle_center is not None else None # Store as numpy array
        self.turning_radius = turning_radius
        # 进度回调函数（GUI 可以传入），用于实时更新进度条
        self.progress_callback = progress_callback
        self._last_progress_percent = -1  # 避免重复回调
        self.integration_method = None # Track which integration method was used
        self.metrics = VehicleMetrics(self.target_trajectory, target_steering_angle) # Pass original target_trajectory here
        self.initialized = True
        self.timing_data = {} # 用于存储计时数据

        # 清除并预分配 NumPy 数组用于存储结果
        self.clear_simulation_data()

    def clear_simulation_data(self):
        """
        清除所有仿真数据，并为 NumPy 数组预分配空间
        """
        # 预分配 NumPy 数组，使用 NaN 作为初始值，更容易发现未赋值的步骤
        # 车辆状态
        self.positions = np.full((self.time_steps, 4), np.nan) # x, y, z, theta
        self.velocities = np.full((self.time_steps, 4), np.nan) # vx, vy, vz, omega
        self.accelerations = np.full((self.time_steps, 4), np.nan) # ax, ay, az, alpha
        # 力、功和力矩
        self.wheel_forces = np.full((self.time_steps, 4, 3), np.nan) # steps, wheels, (fx, fy, fz)
        self.total_forces = np.full((self.time_steps, 3), np.nan) # fx, fy, fz (vehicle total)
        self.wheel_works = np.full((self.time_steps, 4), np.nan) # steps, wheels (power)
        self.wheel_torques = np.full((self.time_steps, 4, 3), np.nan) # steps, wheels (Mx, My, Mz)
        # 车轮绝对位置
        self.wheel_positions = np.full((self.time_steps, 4, 3), np.nan) # steps, wheels, (x, y, z)
        # 指标
        # Ensure metrics.trajectory_errors is pre-allocated correctly if we assign directly
        self.metrics.trajectory_errors = np.full(self.time_steps, np.nan) # Ensure it's a numpy array
        self.metrics.energy_consumptions = np.full(self.time_steps, np.nan)
        self.slip_ratios = np.full(self.time_steps, np.nan) # 平均滑转率

        # 重置其他跟踪变量
        self.integration_method = None # Reset integration method tracker
        self.timing_data = {} # 重置计时数据
        # self.metrics.reset() # VehicleMetrics 没有 reset 方法，改为手动重置

        # 清理每个轮子的历史数据 (如果需要)
        if hasattr(self, 'vehicle') and hasattr(self.vehicle, 'wheels'):
            for wheel in self.vehicle.wheels:
                # 如果 Wheel 类内部有历史记录列表，也需要清空
                if hasattr(wheel, 'omega_history'):
                     wheel.omega_history = [] # 假设是列表

    def generate_trapezoidal_speed(self, max_speed, accel_time, const_time):
        """
        生成梯形速度曲线 (保持不变, 返回 NumPy 数组)
        :param max_speed: 最大速度 (m/s)
        :param accel_time: 加速时间 (s)，如果为0则生成匀速曲线
        :param const_time: 匀速时间 (s)
        :return: 速度曲线数组 (NumPy)
        """
        speed = np.zeros(self.time_steps)
        if accel_time <= 1e-6: # Use tolerance for float comparison
            speed[:] = max_speed
            return speed

        accel_steps = int(np.ceil(accel_time / self.dt))
        const_steps = int(np.ceil(const_time / self.dt))
        # Ensure decel_steps calculation doesn't exceed total steps
        decel_steps = min(accel_steps, self.time_steps - accel_steps - const_steps)
        if decel_steps < 0: decel_steps = 0 # Handle cases where total_time is short

        accel_rate = max_speed / accel_time
        t_accel = np.linspace(0, accel_time, accel_steps, endpoint=False) # Avoid duplicate time point
        end_accel_step = min(accel_steps, self.time_steps)
        speed[:end_accel_step] = accel_rate * t_accel[:end_accel_step]

        start_const_step = end_accel_step
        end_const_step = min(start_const_step + const_steps, self.time_steps)
        speed[start_const_step:end_const_step] = max_speed

        start_decel_step = end_const_step
        end_decel_step = min(start_decel_step + decel_steps, self.time_steps)
        if start_decel_step < self.time_steps and decel_steps > 0:
            # Adjust t_decel length based on available steps
            actual_decel_steps = end_decel_step - start_decel_step
            t_decel = np.linspace(0, accel_time * (actual_decel_steps / decel_steps) if decel_steps > 0 else 0, actual_decel_steps)
            speed[start_decel_step:end_decel_step] = max_speed - accel_rate * t_decel

        speed = np.clip(speed, 0, max_speed)
        # Ensure the array has exactly self.time_steps elements
        if len(speed) > self.time_steps:
            speed = speed[:self.time_steps]
        elif len(speed) < self.time_steps:
            # Pad with the last value if necessary (e.g., if total_time is very short)
            padded_speed = np.full(self.time_steps, speed[-1] if len(speed) > 0 else 0.0)
            padded_speed[:len(speed)] = speed
            speed = padded_speed

        return speed

    def explicit_euler(self, speed_profile, angles=None):
        """
        显式欧拉法仿真 (优化版：使用 NumPy 数组存储)
        :param speed_profile: 速度曲线数组 (time_steps, 4) or (time_steps,)
        :param angles: 转向角度 [左前轮, 右前轮] (rad)
        """
        self.integration_method = 'euler'
        last_position_xy = np.array(self.vehicle.position[:2]) # NumPy array for distance calc

        # Ensure speed_profile has shape (time_steps, 4) if it's per wheel
        if speed_profile.ndim == 1:
             # Assume same speed profile applies to all wheels if 1D
             # This might need adjustment based on how speed_profile is generated/used
             # For steering, it's usually (time_steps, 4) from run_steering_simulation
             print("Warning: explicit_euler received 1D speed_profile. Assuming broadcast needed.")
             # Example: If it should be vehicle speed, not wheel speed:
             # vehicle_speed_profile = speed_profile
             # speeds_input = np.tile(vehicle_speed_profile[:, np.newaxis], (1, 4)) # Placeholder logic
             # Or if it's max_speed profile to be scaled by Ackermann:
             # This case should ideally be handled before calling euler
             speeds_input = speed_profile # Pass as is, let vehicle handle? Risky.
        elif speed_profile.shape == (self.time_steps, 4):
             speeds_input = speed_profile
        else:
             raise ValueError(f"Invalid speed_profile shape: {speed_profile.shape}")

        current_angles = np.array(angles if angles is not None else [0.0, 0.0])

        # --- 主循环 ---
        for t in tqdm(range(self.time_steps), desc="Euler Simulation Progress", unit="step"):
            # 初始化当前步骤的计时字典
            step_timing = {}
            overall_step_start_time = time.perf_counter()

            # 获取当前状态 (减少属性访问)
            current_position = np.array(self.vehicle.position)
            current_velocity = np.array(self.vehicle.velocity)

            # 更新车辆状态 (内部方法调用不可避免)
            start_time = time.perf_counter()
            self.vehicle.input_speed_and_angle(speeds_input[t], current_angles)
            step_timing['input_speed_angle'] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            self.vehicle.update_forces(self.dt)
            step_timing['update_forces'] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            self.vehicle.update_acceleration()
            step_timing['update_acceleration'] = time.perf_counter() - start_time

            # 获取加速度 (减少属性访问)
            current_acceleration = np.array(self.vehicle.acceleration)
            # ax_inertial, ay_inertial, az_body, alpha = current_acceleration

            # 计算新速度 (向量化)
            new_velocity = current_velocity + current_acceleration * self.dt

            # 计算新位置 (向量化)
            # Note: Position update uses the *new* velocity components
            new_position = current_position + new_velocity * self.dt
            # Theta (pos[3]) update is correct as it uses omega (vel[3])

            # 更新车辆对象状态
            self.vehicle.position = new_position.tolist() # Convert back if Vehicle expects list
            self.vehicle.velocity = new_velocity.tolist()

            # 更新车轮状态 (内部方法调用不可避免)
            start_time = time.perf_counter()
            self.vehicle.update_wheel_position()
            step_timing['update_wheel_pos'] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            self.vehicle.update_wheel_velocities()
            step_timing['update_wheel_vel'] = time.perf_counter() - start_time

            # --- 记录结果到 NumPy 数组 ---
            self.positions[t] = new_position
            self.velocities[t] = new_velocity
            self.accelerations[t] = current_acceleration

            # 获取并记录力 (需要循环和方法调用)
            start_time = time.perf_counter()
            step_wheel_forces = np.array([wheel.get_force() for wheel in self.vehicle.wheels]) # (4, 3)
            step_timing['get_wheel_forces'] = time.perf_counter() - start_time
            self.wheel_forces[t] = step_wheel_forces
            self.total_forces[t] = np.array(self.vehicle.total_forces[:3]) # 只取前三个力分量 Fx, Fy, Fz

            # 计算并记录功 (向量化)
            # wheel_velocities_xy = np.array([wheel.velocity[:2] for wheel in self.vehicle.wheels]) # (4, 2)
            # wheel_forces_xy = step_wheel_forces[:, :2] # (4, 2)
            # power_xy = np.einsum('ij,ij->i', wheel_forces_xy, wheel_velocities_xy) # Dot product for each wheel -> (4,)
            # self.wheel_works[t] = power_xy * self.dt

            # 计算并记录力矩，并使用新的方法计算功/能量
            start_time = time.perf_counter()
            torques = np.array([wheel.get_torque() for wheel in self.vehicle.wheels]) # Shape (4, 3)
            step_timing['get_wheel_torques'] = time.perf_counter() - start_time
            self.wheel_torques[t] = torques # Store the full (4, 3) torques for this step

            # 新方法：获取每个车轮基于面元计算的功率
            element_powers = np.array([wheel.element_power for wheel in self.vehicle.wheels]) # Shape (4,)
            self.wheel_works[t] = element_powers # 存储每个车轮的功率

            # 计算总功率 (介质对车轮做功的功率，通常为负)
            step_total_power = np.sum(element_powers)

            # 计算能量消耗 (车辆克服阻力做的功，取绝对值)
            instant_total_energy_consumption = abs(step_total_power) * self.dt
            self.metrics.energy_consumptions[t] = instant_total_energy_consumption


            # 记录车轮位置 (需要循环和方法调用)
            start_time = time.perf_counter()
            self.wheel_positions[t] = np.array([wheel.absolute_position for wheel in self.vehicle.wheels])
            step_timing['get_wheel_abs_pos'] = time.perf_counter() - start_time

            # --- 计算并记录指标 ---
            # 滑转率 (向量化)
            start_time = time.perf_counter()
            v_vehicle_y = abs(new_velocity[1]) # Use updated velocity
            v_rolls = np.array([wheel.get_roll_speed() for wheel in self.vehicle.wheels]) # (4,)
            # Avoid division by zero
            slip_ratios_wheels = np.zeros_like(v_rolls)
            if abs(v_vehicle_y) > 1e-6:
                 slip_ratios_wheels = (v_vehicle_y - v_rolls) / v_vehicle_y
            slip_ratios_wheels = np.clip(slip_ratios_wheels, -1.0, 1.0) # Clip individual ratios
            self.slip_ratios[t] = np.mean(slip_ratios_wheels) # Record average slip
            step_timing['calc_slip_ratio'] = time.perf_counter() - start_time

            # 轨迹误差 (方法调用) -> 改为几何误差计算
            start_time = time.perf_counter()
            current_position_xy = new_position[:2]
            if self.circle_center is not None and self.turning_radius is not None and np.isfinite(self.turning_radius):
                dist_to_center = np.linalg.norm(current_position_xy - self.circle_center)
                instant_error = abs(dist_to_center - self.turning_radius)
                self.metrics.trajectory_errors[t] = instant_error # Assign to numpy array index
            elif self.target_trajectory is not None and len(self.target_trajectory) > 0: # Fallback for straight line or no circle info
                 # Original point-based calculation
                 target_traj_arr = np.array(self.target_trajectory)
                 distances = np.sqrt(np.sum((target_traj_arr - current_position_xy)**2, axis=1))
                 min_distance = np.min(distances)
                 self.metrics.trajectory_errors[t] = min_distance
            else:
                 self.metrics.trajectory_errors[t] = 0.0 # Or np.nan
            # self.metrics.compute_trajectory_error(current_position_xy) # Remove old call
            step_timing['calc_traj_error'] = time.perf_counter() - start_time

            # 行驶距离 (向量化)
            start_time = time.perf_counter()
            distance_increment = np.linalg.norm(current_position_xy - last_position_xy)
            self.metrics.update_distance_travelled(distance_increment)
            step_timing['calc_distance'] = time.perf_counter() - start_time
            last_position_xy = current_position_xy # Update for next step

            # 记录总步长时间
            step_timing['total_step_time'] = time.perf_counter() - overall_step_start_time

            # 将当前步骤的计时数据添加到全局计时数据中
            for key, value in step_timing.items():
                self.timing_data.setdefault(key, []).append(value)

            # --- 进度回调 ---
            if self.progress_callback:
                percent = int(100 * (t + 1) / self.time_steps)
                if percent != self._last_progress_percent:
                    try:
                        self.progress_callback(percent)
                    except Exception:
                        pass  # GUI 回调失败时不影响仿真
                    self._last_progress_percent = percent

    def _calculate_derivatives(self, state, speeds, angles):
        """
        计算状态向量 [x, y, z, theta, vx, vy, vz, omega] 的时间导数 (优化：减少列表转换)
        用于 RK4 积分。会临时修改车辆状态。

        :param state: 当前状态向量 (NumPy array)
        :param speeds: 当前时间步的输入速度 (NumPy array)
        :param angles: 当前时间步的输入转向角 (NumPy array)
        :return: 状态导数向量 (NumPy array)
        """
        # 保存原始状态 (列表)
        original_position = list(self.vehicle.position) # Ensure it's a copy
        original_velocity = list(self.vehicle.velocity) # Ensure it's a copy
        # Consider saving/restoring wheel states if update_forces/accel have side effects beyond position/velocity

        try:
            # 设置车辆状态为 RK4 的中间状态 (直接修改列表)
            for i in range(4): self.vehicle.position[i] = state[i]
            for i in range(4): self.vehicle.velocity[i] = state[i+4]
            # 基于此中间状态更新车轮位置/速度
            self.vehicle.update_wheel_position()
            self.vehicle.update_wheel_velocities()

            # 应用输入并计算此状态下的力/加速度
            self.vehicle.input_speed_and_angle(speeds, angles)
            self.vehicle.update_forces(self.dt)
            self.vehicle.update_acceleration()

            # 获取导数
            # Acceleration is the derivative of velocity
            ax_inertial, ay_inertial, az_body, alpha = self.vehicle.acceleration # Assumes this is updated correctly
            # Velocity is the derivative of position
            vx, vy, vz, omega = state[4:] # Get velocity from the input state

            derivatives = np.array([vx, vy, vz, omega, ax_inertial, ay_inertial, az_body, alpha])

        finally:
            # 恢复原始车辆状态 (恢复列表内容)
            for i in range(4): self.vehicle.position[i] = original_position[i]
            for i in range(4): self.vehicle.velocity[i] = original_velocity[i]
            # 恢复原始车轮状态（如果需要）
            # Re-update wheel states based on original vehicle state
            self.vehicle.update_wheel_position()
            self.vehicle.update_wheel_velocities()

        return derivatives

    def _rk4_step(self, t, current_state, speeds, angles):
        """
        执行单个 RK4 时间步 (使用 NumPy)。
        :param t: 当前时间 (可能在导数计算中未使用)
        :param current_state: 当前状态向量 (NumPy array)
        :param speeds: 此时间步的输入速度 (NumPy array)
        :param angles: 此时间步的输入转向角 (NumPy array)
        :return: 下一个时间步的状态向量 (NumPy array)
        """
        k1 = self._calculate_derivatives(current_state, speeds, angles)
        k2 = self._calculate_derivatives(current_state + 0.5 * self.dt * k1, speeds, angles)
        k3 = self._calculate_derivatives(current_state + 0.5 * self.dt * k2, speeds, angles)
        k4 = self._calculate_derivatives(current_state + self.dt * k3, speeds, angles)

        new_state = current_state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state

    def run_simulation_rk4(self, speed_profile, angles=None):
        """
        使用四阶龙格库塔法运行仿真 (优化版：使用 NumPy 数组存储)。
        :param speed_profile: 速度曲线数组 (time_steps, 4) or (time_steps,)
        :param angles: 转向角度 [左前轮, 右前轮] (rad)
        """
        self.integration_method = 'rk4'
        last_position_xy = np.array(self.vehicle.position[:2]) # NumPy array

        # 初始状态向量 (NumPy array)
        # 初始状态向量 (从车辆列表获取)
        current_state = np.array(self.vehicle.position + self.vehicle.velocity)

        # Ensure speed_profile has shape (time_steps, 4) if it's per wheel
        if speed_profile.ndim == 1:
             print("Warning: run_simulation_rk4 received 1D speed_profile. Assuming broadcast needed.")
             # Adapt as needed, similar to explicit_euler
             speeds_input = speed_profile # Placeholder
        elif speed_profile.shape == (self.time_steps, 4):
             speeds_input = speed_profile
        else:
             raise ValueError(f"Invalid speed_profile shape: {speed_profile.shape}")

        current_angles = np.array(angles if angles is not None else [0.0, 0.0])

        # --- 强制检查和重置 metrics.trajectory_errors ---
        # 理论上不应该需要，但为了解决奇怪的 list assignment error
        if not isinstance(self.metrics.trajectory_errors, np.ndarray) or self.metrics.trajectory_errors.shape != (self.time_steps,):
            print(f"Warning: Resetting metrics.trajectory_errors in run_simulation_rk4. Previous type/shape: {type(self.metrics.trajectory_errors)} / {getattr(self.metrics.trajectory_errors, 'shape', 'N/A')}")
            self.metrics.trajectory_errors = np.full(self.time_steps, np.nan)
        # ----------------------------------------------------

        # --- 主循环 ---
        for t_index in tqdm(range(self.time_steps), desc="RK4 Simulation Progress", unit="step"):
            # 初始化当前步骤的计时字典
            step_timing = {}
            overall_step_start_time = time.perf_counter()

            current_time = self.time[t_index]
            current_speeds = speeds_input[t_index] # Get speeds for this step

            # 执行 RK4 步骤
            start_time = time.perf_counter()
            new_state = self._rk4_step(current_time, current_state, current_speeds, current_angles)
            step_timing['rk4_step'] = time.perf_counter() - start_time

            # 更新车辆对象状态 (更新列表内容)
            for i in range(4): self.vehicle.position[i] = new_state[i]
            for i in range(4): self.vehicle.velocity[i] = new_state[i+4]
            current_state = new_state # Update state for next iteration

            # 基于新状态更新辅助车辆属性（车轮）
            start_time = time.perf_counter()
            self.vehicle.update_wheel_position()
            step_timing['update_wheel_pos_rk4'] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            self.vehicle.update_wheel_velocities()
            step_timing['update_wheel_vel_rk4'] = time.perf_counter() - start_time

            # --- 为记录重新计算加速度和力 ---
            # (RK4 本身不直接给出步末加速度/力，需基于最终状态重新计算)
            start_time = time.perf_counter()
            self.vehicle.input_speed_and_angle(current_speeds, current_angles)
            step_timing['input_speed_angle_rk4'] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            self.vehicle.update_forces(self.dt)
            step_timing['update_forces_rk4'] = time.perf_counter() - start_time

            start_time = time.perf_counter()
            self.vehicle.update_acceleration()
            step_timing['update_acceleration_rk4'] = time.perf_counter() - start_time
            current_acceleration = np.array(self.vehicle.acceleration) # NumPy array

            # --- 记录结果到 NumPy 数组 ---
            self.positions[t_index] = new_state[:4]
            self.velocities[t_index] = new_state[4:]
            self.accelerations[t_index] = current_acceleration

            # 获取并记录力 (需要循环和方法调用)
            start_time = time.perf_counter()
            step_wheel_forces = np.array([wheel.get_force() for wheel in self.vehicle.wheels]) # (4, 3)
            step_timing['get_wheel_forces_rk4'] = time.perf_counter() - start_time
            self.wheel_forces[t_index] = step_wheel_forces
            self.total_forces[t_index] = np.array(self.vehicle.total_forces)[:3] # 只取前3个力分量 (Fx, Fy, Fz)

            # 计算并记录力矩，并使用新的方法计算功/能量
            start_time = time.perf_counter()
            torques = np.array([wheel.get_torque() for wheel in self.vehicle.wheels]) # Shape (4, 3)
            step_timing['get_wheel_torques_rk4'] = time.perf_counter() - start_time
            self.wheel_torques[t_index] = torques # Store the full (4, 3) torques for this step

            # 新方法：获取每个车轮基于面元计算的功率
            element_powers = np.array([wheel.element_power for wheel in self.vehicle.wheels]) # Shape (4,)
            self.wheel_works[t_index] = element_powers # 存储每个车轮的功率

            # 计算总功率 (介质对车轮做功的功率，通常为负)
            step_total_power = np.sum(element_powers)

            # 计算能量消耗 (车辆克服阻力做的功，取绝对值)
            instant_total_energy_consumption = abs(step_total_power) * self.dt
            self.metrics.energy_consumptions[t_index] = instant_total_energy_consumption

            # 记录车轮位置 (需要循环和方法调用)
            start_time = time.perf_counter()
            self.wheel_positions[t_index] = np.array([wheel.absolute_position for wheel in self.vehicle.wheels])
            step_timing['get_wheel_abs_pos_rk4'] = time.perf_counter() - start_time

            # --- 计算并记录指标 ---
            # 滑转率 (向量化)
            start_time = time.perf_counter()
            v_vehicle_y = abs(new_state[5]) # Use updated velocity y (index 5 in state)
            v_rolls = np.array([wheel.get_roll_speed() for wheel in self.vehicle.wheels]) # (4,)
            slip_ratios_wheels = np.zeros_like(v_rolls)
            if abs(v_vehicle_y) > 1e-6:
                 slip_ratios_wheels = (v_vehicle_y - v_rolls) / v_vehicle_y
            slip_ratios_wheels = np.clip(slip_ratios_wheels, -1.0, 1.0)
            self.slip_ratios[t_index] = np.mean(slip_ratios_wheels)
            step_timing['calc_slip_ratio_rk4'] = time.perf_counter() - start_time

            # 轨迹误差 (直接计算并赋值) -> 改为几何误差计算
            start_time = time.perf_counter()
            current_position_xy = new_state[:2]
            if self.circle_center is not None and self.turning_radius is not None and np.isfinite(self.turning_radius):
                dist_to_center = np.linalg.norm(current_position_xy - self.circle_center)
                instant_error = abs(dist_to_center - self.turning_radius)
                self.metrics.trajectory_errors[t_index] = instant_error
            elif self.target_trajectory is not None and len(self.target_trajectory) > 0: # Fallback for straight line or no circle info
                 # Original point-based calculation (already here)
                 target_traj_arr = np.array(self.target_trajectory)
                 distances = np.sqrt(np.sum((target_traj_arr - current_position_xy)**2, axis=1))
                 min_distance = np.min(distances)
                 self.metrics.trajectory_errors[t_index] = min_distance
            else:
                 self.metrics.trajectory_errors[t_index] = 0.0 # 或者 np.nan
            step_timing['calc_traj_error_rk4'] = time.perf_counter() - start_time

            # 行驶距离 (向量化)
            start_time = time.perf_counter()
            distance_increment = np.linalg.norm(current_position_xy - last_position_xy)
            self.metrics.update_distance_travelled(distance_increment)
            step_timing['calc_distance_rk4'] = time.perf_counter() - start_time
            last_position_xy = current_position_xy # Update for next step

            # 记录总步长时间
            step_timing['total_step_time_rk4'] = time.perf_counter() - overall_step_start_time

            # 将当前步骤的计时数据添加到全局计时数据中
            for key, value in step_timing.items():
                self.timing_data.setdefault(key, []).append(value)

            # --- 进度回调 ---
            if self.progress_callback:
                percent = int(100 * (t_index + 1) / self.time_steps)
                if percent != self._last_progress_percent:
                    try:
                        self.progress_callback(percent)
                    except Exception:
                        pass
                    self._last_progress_percent = percent

        print("RK4 simulation finished.")
        # 绘图和返回结果由调用函数处理

    def run_steering_simulation(self, max_speed, accel_time, const_time, steering_angle,ackermann_ratio,speed_changed=True,angle_changed=True):
        """
        运行转向仿真 (调用优化后的 Euler 或 RK4)
        """
        self.clear_simulation_data() # Resets and pre-allocates arrays
        from Ackermann import Ackermann # Local import ok

        # --- 参数计算 (保持不变) ---
        wheel_positions_local = np.array([wheel.position for wheel in self.vehicle.wheels]) # Use NumPy
        track_width = abs(wheel_positions_local[0, 0] - wheel_positions_local[1, 0])
        wheelbase = abs(wheel_positions_local[0, 1] - wheel_positions_local[2, 1])
        ackermann = Ackermann(track_width, wheelbase)

        if angle_changed:
            # Ackermann returns angles in radians
            left_steer, right_steer = ackermann.calculate_steering_angles(steering_angle, ackermann_ratio)
        else:
            left_steer = right_steer = 0.0

        if speed_changed:
            # Ackermann returns speeds in m/s
            v_fl, v_fr, v_rl, v_rr = ackermann.calculate_wheel_speeds(steering_angle, max_speed)
        else:
            v_fl = v_fr = v_rl = v_rr = max_speed

        # --- 保存仿真参数 (保持不变) ---
        self.simulation_params = {
            'max_speed': max_speed, 'accel_time': accel_time, 'const_time': const_time,
            'steering_angle': steering_angle, 'ackermann_ratio': ackermann_ratio,
            'total_time': self.total_time, 'time_step': self.dt,
            'left_steer_rad': left_steer, 'right_steer_rad': right_steer,
            'target_wheel_speeds': [v_fl, v_fr, v_rl, v_rr]
        }
        # print("Starting steering simulation with parameters:", self.simulation_params) # Optional debug

        # --- 生成速度曲线 ---
        # generate_trapezoidal_speed returns the profile for the *vehicle's target speed*
        vehicle_speed_profile = self.generate_trapezoidal_speed(max_speed, accel_time, const_time)
        # Scale this profile for each wheel based on Ackermann speeds
        # Avoid division by zero if max_speed is zero
        if max_speed > 1e-6:
             normalized_profile = vehicle_speed_profile / max_speed
        else:
             normalized_profile = np.zeros_like(vehicle_speed_profile)

        target_wheel_speeds = np.array([v_fl, v_fr, v_rl, v_rr])
        # speed_profile shape: (time_steps, 4)
        speed_profile_input = np.outer(normalized_profile, target_wheel_speeds)

        # --- 运行仿真 ---
        if INTEGRATION_METHOD == 'rk4':
            self.run_simulation_rk4(speed_profile_input, angles=[left_steer, right_steer])
        else:
            self.explicit_euler(speed_profile_input, angles=[left_steer, right_steer])
        print("\nSimulation completed successfully!")

        # --- 处理结果 (绘图/保存/返回) ---
        # plot_and_save_results 现在会使用 NumPy 数组
        results = self.plot_and_save_results()

        # 从 metrics 获取最终指标
        avg_trajectory_error, total_energy_consumptions = self.metrics.get_average_metrics()
        total_distance_travelled = self.metrics.distance_travelled

        # 添加到返回字典 (确保 metrics 计算正确)
        results['avg_trajectory_error'] = avg_trajectory_error
        results['total_energy_consumptions'] = total_energy_consumptions
        results['total_distance_travelled'] = total_distance_travelled
        # results['wheel_speeds_history'] = self.velocities # Example if needed
        results['timing_data'] = self.timing_data # 添加计时数据

        return results


    def run_steering_optimization_simulation(self, max_speed, accel_time, const_time,
                                         left_steering_angle, right_steering_angle,
                                         beta_FR, beta_RL, beta_RR,
                                         speed_changed=True, angle_changed=True):
        """
        为优化设计的转向仿真函数 (调用优化后的 Euler 或 RK4)
        直接接受优化参数（绝对转向角度）。
        """
        self.clear_simulation_data() # Resets and pre-allocates arrays
        # from Ackermann import Ackermann # Not needed if ratios are given directly

        # --- 参数计算 ---
        # 使用传入的绝对转向角度 (单位: 度)
        # 仿真内部使用弧度，并遵循车辆模型约定 (例如，左转为负)
        left_steer = -np.radians(left_steering_angle)
        right_steer = -np.radians(right_steering_angle)

        # Apply speed scaling factors
        v_fl = max_speed # Assume FL is reference or use another logic
        v_fr = max_speed * beta_FR
        v_rl = max_speed * beta_RL
        v_rr = max_speed * beta_RR
        target_wheel_speeds = np.array([v_fl, v_fr, v_rl, v_rr])

        # --- 保存仿真参数 ---
        self.simulation_params = {
            'max_speed': max_speed, 'accel_time': accel_time, 'const_time': const_time,
            'left_steering_angle_deg': left_steering_angle,
            'right_steering_angle_deg': right_steering_angle,
            'beta_FR': beta_FR, 'beta_RL': beta_RL, 'beta_RR': beta_RR,
            'total_time': self.total_time, 'time_step': self.dt,
            'left_steer_rad': left_steer, 'right_steer_rad': right_steer,
            'target_wheel_speeds': target_wheel_speeds.tolist()
        }

        # --- 生成速度曲线 ---
        vehicle_speed_profile = self.generate_trapezoidal_speed(max_speed, accel_time, const_time)
        if max_speed > 1e-6:
             normalized_profile = vehicle_speed_profile / max_speed
        else:
             normalized_profile = np.zeros_like(vehicle_speed_profile)
        # speed_profile shape: (time_steps, 4)
        speed_profile_input = np.outer(normalized_profile, target_wheel_speeds)

        # --- 运行仿真 ---
        if INTEGRATION_METHOD == 'rk4':
            self.run_simulation_rk4(speed_profile_input, angles=[left_steer, right_steer])
        else:
            self.explicit_euler(speed_profile_input, angles=[left_steer, right_steer])

        # --- 处理结果 ---
        # plot_and_save_results uses the populated NumPy arrays
        results = self.plot_and_save_results()

        # 从 metrics 获取最终指标
        avg_trajectory_error, total_energy_consumptions = self.metrics.get_average_metrics()
        total_distance_travelled = self.metrics.distance_travelled

        # 添加到返回字典
        results['avg_trajectory_error'] = avg_trajectory_error
        results['total_energy_consumptions'] = total_energy_consumptions
        results['total_distance_travelled'] = total_distance_travelled
        # Add optimization-specific params if needed
        results['input_params'] = self.simulation_params
        results['timing_data'] = self.timing_data # 添加计时数据

        return results


    def plot_and_save_results(self, output_dir=None):
        """
        绘制并保存仿真结果 (使用优化后的 NumPy 数组)
        :param output_dir: 保存图片的目录，如果为 None 则不保存
        :return: 包含绘图文件路径的字典 (如果保存)
        """
        plotter = Plotter() # 获取单例实例，不传递数据
        saved_plots = {}
        # 直接调用绘图方法，将 plot_mode 传递给 Plotter 处理
        if self.plot_mode != 'none': # 只有在需要绘图或保存时才调用
            print("\nGenerating plots...")
            try:
                # 调用 plotter 的方法，并将 simulator 实例 (self) 传递给它
                # 假设 plot_simulation_results 包含了所有绘图/保存逻辑并返回保存路径字典
                saved_plots_result = plotter.plot_simulation_results(self, plot_mode=self.plot_mode, output_dir=output_dir)
                # 确保 saved_plots_result 是字典
                if isinstance(saved_plots_result, dict):
                    saved_plots = saved_plots_result
                else:
                    # 如果返回的不是字典 (例如 None 或其他)，则保持 saved_plots 为空字典
                    print("Warning: plot_simulation_results did not return a dictionary of saved paths.")
                    saved_plots = {}

            except Exception as e:
                print(f"Error during plotting/saving: {e}")
                # 即使绘图失败，也继续执行并返回空字典
                saved_plots = {} # 确保返回的是字典

        # 返回包含结果的字典
        # 注意：如果 plot_mode 是 'none'，saved_plots 将是 {}
        results_summary = {
             "plot_files": saved_plots
             # 可以添加其他仿真总结性结果到这里
        }
        return results_summary
