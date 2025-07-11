import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import os
import numpy as np

def filter_y_data(y_data):
    """
    对y轴数据进行滤波，当数据小于-1时，强制设为-1。
    """
    y_data = np.clip(y_data, a_min=-1, a_max=None)
    return y_data

class Plotter:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Plotter, cls).__new__(cls)
        return cls._instance

    def calculate_min_distance_to_circle(self, traj, circle_center, radius):
        """计算轨迹点到圆的最小距离"""
        x, y = traj['x'], traj['y']
        cx, cy = circle_center
        distances = np.sqrt((x - cx)**2 + (y - cy)**2) - radius
        return np.abs(distances)

    def plot_trajectories(self, trajectories, labels=None, colors=None):
        """绘制多条轨迹"""
        if labels is None:
            labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]
        if colors is None:
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:len(trajectories)]

        plt.figure(figsize=(10, 6))
        for traj, label, color in zip(trajectories, labels, colors):
            plt.plot(traj['x'], traj['y'], label=label, color=color)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Trajectory Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_distance_over_time(self, trajectories, labels=None, colors=None):
        """绘制轨迹点到理想圆的最小距离随时间变化"""
        if labels is None:
            labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]
        if colors is None:
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:len(trajectories)]

        # 假设第一个轨迹是理想轨迹
        ideal_traj = trajectories[0]
        
        # 计算理想轨迹的圆
        x_center = ideal_traj['x'].mean()
        y_center = ideal_traj['y'].mean()
        radius = np.mean(np.sqrt((ideal_traj['x'] - x_center)**2 + (ideal_traj['y'] - y_center)**2))
        circle_center = (x_center, y_center)
        
        plt.figure(figsize=(10, 6))
        for traj, label, color in zip(trajectories, labels, colors):
            distances = self.calculate_min_distance_to_circle(traj, circle_center, radius)
            plt.plot(traj['time'], distances, label=label, color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance to Ideal Circle (m)')
        plt.title('Distance to Ideal Trajectory Circle Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_trajectory_comparison_with_distance(self, simulator, output_dir=None):
        """
        绘制目标轨迹、实际轨迹和最小距离随时间变化的图
        :param simulator: Simulator对象
        :param output_dir: 输出目录路径
        """
        # 准备数据
        positions = np.array(simulator.positions)
        time = simulator.time
        
        # 创建DataFrame
        df = pd.DataFrame({
            'time': time,
            'x': positions[:, 0],
            'y': positions[:, 1]
        })
        
        # 如果有目标轨迹，也转换为DataFrame
        if simulator.target_trajectory and len(simulator.target_trajectory) > 0:
            target_x = [p[0] for p in simulator.target_trajectory]
            target_y = [p[1] for p in simulator.target_trajectory]
            target_time = np.linspace(0, time[-1], len(target_x))
            
            target_df = pd.DataFrame({
                'time': target_time,
                'x': target_x,
                'y': target_y
            })
            
            trajectories = [target_df, df]
            labels = ['Target Trajectory', 'Actual Trajectory']
            colors = ['red', 'blue']
            
            # 绘制轨迹
            # self.plot_trajectories(trajectories, labels, colors)
            # plt.close()
            
            # 绘制距离随时间变化
            self.plot_distance_over_time(trajectories, labels, colors)
            plt.close()

    def plot_trajectory_comparison(self, simulator, output_dir=None):
        """
        原有的轨迹对比方法，保持不变
        """
        positions = np.array(simulator.positions)
        
        plt.figure(figsize=(10, 10))
        
        # 绘制目标轨迹
        if simulator.target_trajectory and len(simulator.target_trajectory) > 0:
            target_x = [p[0] for p in simulator.target_trajectory]
            target_y = [p[1] for p in simulator.target_trajectory]
            plt.plot(target_x, target_y, 'r--', label='Target Trajectory', linewidth=2)
        
        # 绘制实际轨迹
        plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Actual Trajectory', linewidth=2)
        
        plt.legend()
        plt.title('Trajectory Comparison')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f'output/{timestamp}')
            output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'trajectory_comparison.png')
        plt.close()

    def plot_simulation_results(self, simulator, timestamps, plot_mode='show', output_dir=None):
        """
        绘制仿真结果并保存数据。
        :param simulator: Simulator 对象
        :param timestamps: 与 simulator 中数据对齐的时间戳数组 (可能被截断)
        :param plot_mode: 'show', 'save', 'energy', 'slip', 'none'
        :param output_dir: 保存图像的目录
        如果 plot_mode == 'save'，则返回包含保存文件路径的字典。
        """
        from pathlib import Path
        if output_dir is not None and not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        saved_paths = {} # 用于存储保存的文件路径
        time = timestamps # 使用传入的时间戳数组
        # 确保传入的时间戳长度与 simulator 中的数据长度一致 (理论上应该一致，除非调用逻辑错误)
        n_steps = len(time)
        positions = np.array(simulator.positions)[:n_steps] # 使用切片确保长度匹配
        velocities = np.array(simulator.velocities)[:n_steps]
        # wheel_speeds 依赖 omega_history，可能不准确或不存在，保持注释
        # wheel_speeds = np.array([[omega[0] for omega in wheel.omega_history] for wheel in simulator.vehicle.wheels]).T[:n_steps]

        if plot_mode == 'energy':  # --- 添加能量消耗绘图 ---
            plt.figure(figsize=(10, 6))
            energy_consumptions = np.array(simulator.metrics.energy_consumptions)[:n_steps]
            plt.plot(time, energy_consumptions, label='Energy Consumption')
            plt.xlabel('Time (s)')
            plt.ylabel('Energy Consumption (J)')
            plt.title('Energy Consumption Over Time')
            plt.legend()
            plt.grid(True)
            
            plt.show()
            # 保存逻辑 (如果需要)
            if plot_mode == 'save':
                if output_dir is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = Path(f'output/{timestamp}') # 使用 data 或 output 目录?
                    output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / 'energy_consumption.png'
                plt.savefig(save_path)
                saved_paths['energy'] = str(save_path)
                print(f"- Saved: {save_path}")
            # 关闭图形
            plt.close()

        elif plot_mode == 'slip':   # --- 添加滑转率绘图 ---
            plt.figure(figsize=(10, 6))
            # 对slip_ratios进行滤波处理
            slip_ratios = np.array(simulator.slip_ratios)[:n_steps]
            filtered_slip_ratios = filter_y_data(slip_ratios)
            plt.plot(time, filtered_slip_ratios, label='Average Slip Ratio')
            plt.xlabel('Time (s)')
            plt.ylabel('Slip Ratio')
            plt.title('Slip Ratio Over Time')
            plt.legend()
            plt.grid(True)
            plt.show()
            # 保存逻辑 (如果需要)
            if plot_mode == 'save':
                if output_dir is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = Path(f'output/{timestamp}')
                    output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / 'slip_ratio.png'
                plt.savefig(save_path)
                saved_paths['slip'] = str(save_path)
                print(f"- Saved: {save_path}")
            # 关闭图形
            plt.close()

        elif plot_mode == 'show' or plot_mode == 'save':
            # 绘制第一组图 (3x3)
            plt.figure(figsize=(18, 12))
            
            # 1. 车辆和轮子XY轨迹，以及目标轨迹
            plt.subplot(3, 3, 1)
            # 优化：先绘制目标轨迹，再绘制车辆轨迹，最后绘制四个轮胎轨迹，颜色和线型区分
            wheel_colors = ['#FF7F0E', '#2CA02C', '#1F77B4', '#D62728']  # 橙、绿、蓝、红
            wheel_styles = ['-', '--', '-.', ':']  # 四种线型

            # 不再绘制目标轨迹

            # 只绘制车辆轨迹（中间层）
            plt.plot(positions[:, 0], positions[:, 1], color='blue', linestyle='-', label='Robot Trajectory', linewidth=2, zorder=2)

            # 3. 绘制四个轮胎轨迹（最上层，颜色和线型区分）
            wheel_positions = np.array(simulator.wheel_positions)[:n_steps]  # 切片确保长度
            for wheel_idx in range(4):
                plt.plot(
                    wheel_positions[:, wheel_idx, 0],
                    wheel_positions[:, wheel_idx, 1],
                    color=wheel_colors[wheel_idx % len(wheel_colors)],
                    linestyle=wheel_styles[wheel_idx % len(wheel_styles)],
                    label=f'Wheel {wheel_idx+1} Trajectory',
                    linewidth=1.8,
                    zorder=3+wheel_idx
                )

            plt.legend()
            plt.title('Vehicle and Wheel XY Trajectories')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.axis('equal')
            plt.grid(True)
        
            # 2. 车辆YZ轨迹
            plt.subplot(3, 3, 2)
            plt.plot(positions[:, 1], positions[:, 2])
            plt.title('Vehicle YZ Trajectory')
            plt.xlabel('Y Position (m)')
            plt.ylabel('Z Position (m)')
            plt.grid(True)
        
            # 3. 三个方向速度
            plt.subplot(3, 3, 3)
            plt.plot(time, velocities[:, 0], label='Vx')
            plt.plot(time, velocities[:, 1], label='Vy')
            plt.plot(time, velocities[:, 2], label='Vz')
            plt.title('Velocities vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (m/s)')
            plt.legend()
            plt.grid(True)
        
            # 4. 角速度
            plt.subplot(3, 3, 4)
            plt.plot(time, velocities[:, 3])
            plt.title('Angular Velocity vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Angular Velocity (rad/s)')
            plt.grid(True)
        
            # 5. 每一个轮子的绕x轴的轮速
            plt.subplot(3, 3, 5)
            for i in range(4):
                # Adjust plotting based on integration method
                # 移除 integration_method 相关逻辑
                # 注意：wheel_speeds 的数据来源可能需要调整，因为它依赖于 simulator.vehicle.wheels[i].omega_history
                # 在优化后的 simulator 中，omega_history 可能不再填充或已被移除
                # 暂时注释掉轮速绘制，如果需要，需要从 simulator.velocities 或其他地方重新计算轮速
                # plt.plot(time, wheel_speeds[:, i], label=f'Wheel {i+1}')
                pass # 暂时不绘制轮速
            plt.title('Wheel Speeds vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Wheel Speed (rad/s)')
            # plt.legend() # 注释掉，因为没有绘制带标签的线条
            plt.grid(True)

            # 6. 位置x随时间变化
            plt.subplot(3, 3, 6)
            plt.plot(time, positions[:, 0])
            plt.title('X Position vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('X Position (m)')
            plt.grid(True)
        
            # 7. 位置y随时间变化
            plt.subplot(3, 3, 7)
            plt.plot(time, positions[:, 1])
            plt.title('Y Position vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Y Position (m)')
            plt.grid(True)
        
            # 8. 位置z随时间变化
            plt.subplot(3, 3, 8)
            plt.plot(time, positions[:, 2])
            plt.title('Z Position vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Z Position (m)')
            plt.grid(True)
        
            # 9. 转角随时间变化
            plt.subplot(3, 3, 9)
            plt.plot(time, positions[:, 3])
            plt.title('Yaw Angle vs Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Yaw Angle (rad)')
            plt.grid(True)
        
            # 调整布局并显示
            plt.tight_layout()
            if plot_mode == 'show':
                plt.show()
            elif plot_mode == 'save':
                if output_dir is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = Path(f'output/{timestamp}')
                    output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / 'trajectory_and_velocity.png'
                plt.savefig(save_path)
                saved_paths['trajectory_velocity'] = str(save_path)
                print(f"- Saved: {save_path}")
                plt.close()

            # 绘制第二组图 (3x3)
            plt.figure(figsize=(18, 18))
        
            # 1. 车体xyz合力
            plt.subplot(3, 3, 1)
            total_forces = np.array(simulator.total_forces)[:n_steps] # 切片确保长度
            plt.plot(time, total_forces[:, 0], label='Fx')
            plt.plot(time, total_forces[:, 1], label='Fy')
            plt.plot(time, total_forces[:, 2], label='Fz')
            plt.title('Vehicle Total Forces')
            plt.xlabel('Time (s)')
            plt.ylabel('Force (N)')
            plt.legend()
            plt.grid(True)
        
            # 2. 车体旋转力矩
            plt.subplot(3, 3, 2)
            if total_forces.shape[1] > 3:
                plt.plot(time, total_forces[:, 3], label='Tz')
                plt.title('Vehicle Torque')
                plt.xlabel('Time (s)')
                plt.ylabel('Torque (Nm)')
                plt.legend()
            else:
                plt.title('Vehicle Torque (No Tz Data)')
                plt.xlabel('Time (s)')
                plt.ylabel('Torque (Nm)')
                plt.text(0.5, 0.5, 'No Tz data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.grid(True)
        
            # 3. 轮子XY轨迹图
            plt.subplot(3, 3, 3)
            # wheel_positions 已在上面获取并切片
            for wheel_idx in range(4):
                plt.plot(wheel_positions[:, wheel_idx, 0],
                        wheel_positions[:, wheel_idx, 1],
                        label=f'Wheel {wheel_idx+1}')
            plt.title('Wheel XY Trajectories')
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
        
            # 4. 轮子X位置随时间变化
            plt.subplot(3, 3, 4)
            for wheel_idx in range(4):
                plt.plot(time, wheel_positions[:, wheel_idx, 0], 
                        label=f'Wheel {wheel_idx+1}')
            plt.title('Wheel X Positions')
            plt.xlabel('Time (s)')
            plt.ylabel('X Position (m)')
            plt.legend()
            plt.grid(True)
        
            # 5. 轮子Y位置随时间变化
            plt.subplot(3, 3, 5)
            for wheel_idx in range(4):
                plt.plot(time, wheel_positions[:, wheel_idx, 1], 
                        label=f'Wheel {wheel_idx+1}')
            plt.title('Wheel Y Positions')
            plt.xlabel('Time (s)')
            plt.ylabel('Y Position (m)')
            plt.legend()
            plt.grid(True)
        
            # 6. 轮子Z位置随时间变化
            plt.subplot(3, 3, 6)
            for wheel_idx in range(4):
                plt.plot(time, wheel_positions[:, wheel_idx, 2], 
                        label=f'Wheel {wheel_idx+1}')
            plt.title('Wheel Z Positions')
            plt.xlabel('Time (s)')
            plt.ylabel('Z Position (m)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            if plot_mode == 'show':
                plt.show()
            elif plot_mode == 'save':
                save_path = output_dir / 'forces_and_wheel_positions.png'
                plt.savefig(save_path)
                saved_paths['forces_wheel_pos'] = str(save_path)
                print(f"- Saved: {save_path}")
                plt.close()

            # Create a third figure for wheel forces
            plt.figure(figsize=(18, 12))
        
            # Plot wheel forces in 2x2 grid
            wheel_forces = np.array(simulator.wheel_forces)[:n_steps] # 切片确保长度
            for wheel_idx in range(4):
                plt.subplot(2, 2, wheel_idx+1)
                plt.plot(time, wheel_forces[:, wheel_idx, 2], label='Fz', color='b')
                plt.title(f'Wheel {wheel_idx+1} Fz')
                plt.xlabel('Time (s)')
                plt.ylabel('Fz (N)')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            if plot_mode == 'show':
                plt.show()
            elif plot_mode == 'save':
                save_path = output_dir / 'wheel_forces.png'
                plt.savefig(save_path)
                saved_paths['wheel_forces'] = str(save_path)
                print(f"- Saved: {save_path}")
                plt.close()

        # 绘制轮胎X轴力（Fx）
        plt.figure(figsize=(18, 12))
        for wheel_idx in range(4):
            plt.subplot(2, 2, wheel_idx+1)
            plt.plot(time, wheel_forces[:, wheel_idx, 0], label='Fx', color='r')
            plt.title(f'Wheel {wheel_idx+1} Fx')
            plt.xlabel('Time (s)')
            plt.ylabel('Fx (N)')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        if plot_mode == 'show':
            plt.show()
        elif plot_mode == 'save':
            save_path = output_dir / 'wheel_forces_X.png'
            plt.savefig(save_path)
            saved_paths['wheel_forces_X'] = str(save_path)
            print(f"- Saved: {save_path}")
        plt.close()

        # 绘制轮胎Y轴力（Fy）
        plt.figure(figsize=(18, 12))
        for wheel_idx in range(4):
            plt.subplot(2, 2, wheel_idx+1)
            plt.plot(time, wheel_forces[:, wheel_idx, 1], label='Fy', color='g')
            plt.title(f'Wheel {wheel_idx+1} Fy')
            plt.xlabel('Time (s)')
            plt.ylabel('Fy (N)')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        if plot_mode == 'show':
            plt.show()
        elif plot_mode == 'save':
            save_path = output_dir / 'wheel_forces_Y.png'
            plt.savefig(save_path)
            saved_paths['wheel_forces_Y'] = str(save_path)
            print(f"- Saved: {save_path}")
        plt.close()

        # 绘制轮胎力矩（Torque）
        plt.figure(figsize=(18, 12))
        # 检查 simulator 对象是否有 wheel_torques 属性
        if hasattr(simulator, 'wheel_torques'):
            wheel_torques = np.array(simulator.wheel_torques)[:n_steps] # 切片确保长度
            # 检查数组维度是否正确 (现在使用 n_steps 而不是 len(time))
            if wheel_torques.ndim == 3 and wheel_torques.shape[0] == n_steps and wheel_torques.shape[1] == 4 and wheel_torques.shape[2] >= 3:
                # 绘制 Mz 力矩 (假设是第3个分量，索引为2)
                torque_component_index = 2
                torque_component_label = 'Mz'
                for wheel_idx in range(4):
                    plt.subplot(2, 2, wheel_idx+1)
                    plt.plot(time, wheel_torques[:, wheel_idx, torque_component_index], label=torque_component_label, color='purple') # 使用紫色区分
                    plt.title(f'Wheel {wheel_idx+1} Torque ({torque_component_label})')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Torque (Nm)')
                    plt.legend()
                    plt.grid(True)
            else:
                # 如果数据无效，在所有子图中显示提示
                for wheel_idx in range(4):
                    plt.subplot(2, 2, wheel_idx+1)
                    plt.title(f'Wheel {wheel_idx+1} Torque (Invalid Data)')
                    plt.text(0.5, 0.5, 'Invalid Torque Data Shape', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.grid(True)
        else:
            # 如果没有力矩数据，在所有子图中显示提示
            for wheel_idx in range(4):
                plt.subplot(2, 2, wheel_idx+1)
                plt.title(f'Wheel {wheel_idx+1} Torque (No Data)')
                plt.text(0.5, 0.5, 'No Torque Data Available', ha='center', va='center', transform=plt.gca().transAxes)
                plt.grid(True)

        plt.tight_layout()
        if plot_mode == 'show':
            plt.show()
        elif plot_mode == 'save':
            # 确保 output_dir 已定义
            if output_dir is None:
                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                 output_dir = Path(f'output/{timestamp}')
                 output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / 'wheel_torques.png'
            plt.savefig(save_path)
            saved_paths['wheel_torques'] = str(save_path) # 添加到返回字典
            print(f"- Saved: {save_path}")
        plt.close()


        # 在函数末尾返回保存路径的字典
        return saved_paths

    def save_simulation_data(self, simulator, output_dir=None, steering_angle=None, ratio=None, differential=None):
        """
        保存仿真数据到CSV文件
        """
        # 生成文件名前缀
        prefix_parts = []
        if steering_angle is not None:
            prefix_parts.append(f"steering_{steering_angle}")
        if ratio is not None:
            prefix_parts.append(f"ratio_{ratio}")
        if differential is not None:
            prefix_parts.append("diff" if differential else "no_diff")
            
        prefix = "_".join(prefix_parts) + "_" if prefix_parts else ""
        
        # Use provided output_dir or create default one
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f'data/{prefix}{timestamp}')
            output_dir.mkdir(parents=True, exist_ok=True)
        vehicle_data_path = output_dir / 'vehicle_data.csv'
        wheel_data_path = output_dir / 'wheel_data.csv'
        
        # 验证数据长度一致性
        data_lengths = {
            'time': len(simulator.time),
            'positions': len(simulator.positions),
            'velocities': len(simulator.velocities),
            'accelerations': len(simulator.accelerations)
        }
        
        if len(set(data_lengths.values())) != 1:
            raise ValueError(f"Data length mismatch: {data_lengths}")
            
        vehicle_data = pd.DataFrame({
            'time': simulator.time,
            'x': np.array(simulator.positions)[:, 0],
            'y': np.array(simulator.positions)[:, 1],
            'z': np.array(simulator.positions)[:, 2],
            'vx': np.array(simulator.velocities)[:, 0],
            'vy': np.array(simulator.velocities)[:, 1],
            'vz': np.array(simulator.velocities)[:, 2],
            'ax': np.array(simulator.accelerations)[:, 0],
            'ay': np.array(simulator.accelerations)[:, 1],
            'az': np.array(simulator.accelerations)[:, 2]
        })
        vehicle_data.to_csv(vehicle_data_path, index=False)
        
        # 保存轮子受力数据和做功数据
        wheel_data_frames = []
        wheel_positions_arr = np.array(simulator.wheel_positions)
        wheel_works_arr = np.array(simulator.wheel_works)
        wheel_forces_arr = np.array(simulator.wheel_forces)
        
        for wheel_idx in range(4):
            wheel_data = pd.DataFrame({
                'time': simulator.time,
                'x': wheel_positions_arr[:, wheel_idx, 0],
                'y': wheel_positions_arr[:, wheel_idx, 1],
                'z': wheel_positions_arr[:, wheel_idx, 2],
                'fx': wheel_forces_arr[:, wheel_idx, 0],
                'fy': wheel_forces_arr[:, wheel_idx, 1],
                'fz': wheel_forces_arr[:, wheel_idx, 2],
                'work': wheel_works_arr[:, wheel_idx]
            })
            wheel_data_frames.append(wheel_data)
        
        wheel_data = pd.concat(wheel_data_frames, keys=range(4), names=['wheel', 'index'])
        wheel_data.to_csv(wheel_data_path, index=True)
        
        return {
            'vehicle_data_path': vehicle_data_path,
            'wheel_data_path': wheel_data_path,
            'output_dir': output_dir
        }
