# vehicle_metrics.py

import numpy as np

class VehicleMetrics:
    def __init__(self, target_trajectory=None, target_steering_angle=0.0):
        """
        初始化性能计算类
        :param target_trajectory: 目标轨迹，格式为[(x1, y1), (x2, y2), ...]，默认为None
        :param target_steering_angle: 目标转向角度，默认为0.0
        """
        self.target_trajectory = target_trajectory if target_trajectory is not None else []
        self.target_steering_angle = target_steering_angle
        self.trajectory_errors = []  # 用于记录每个时间步的轨迹误差
        self.energy_consumptions = []  # 用于记录每个时间步的能量消耗
        self.distance_travelled = 0.0  # 用于记录运行里程


    def compute_trajectory_error(self, current_position):
        """
        计算轨迹误差
        :param current_position: 当前车辆的位置，格式为(x, y)
        :return: 当前轨迹误差
        """
        if not self.target_trajectory:
            return 0.0  # 如果没有目标轨迹，误差为0

        # 计算目标轨迹上的最近点
        min_distance = float('inf')
        closest_point = None
        for target_point in self.target_trajectory:
            distance = np.sqrt((current_position[0] - target_point[0]) ** 2 + (current_position[1] - target_point[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_point = target_point

        self.trajectory_errors.append(min_distance)
        return min_distance

    def update_distance_travelled(self, distance_increment):
        """
        更新行驶里程
        :param distance_increment: 本次迭代中车辆行驶的距离增量
        """
        self.distance_travelled += distance_increment

    def get_average_metrics(self):
        
        # 计算平均轨迹误差 (处理 NumPy 数组和可能的 NaN)
        errors_array = np.array(self.trajectory_errors)
        valid_errors = errors_array[~np.isnan(errors_array)]
        if len(valid_errors) > 0:
            avg_trajectory_error = np.mean(valid_errors)
        else:
            avg_trajectory_error = 0.0 # 或者 np.nan

        # 计算总能量消耗 (处理 NumPy 数组和可能的 NaN)
        energy_array = np.array(self.energy_consumptions)
        valid_energy = energy_array[~np.isnan(energy_array)]
        total_energy_consumption = np.sum(valid_energy)

        return avg_trajectory_error, total_energy_consumption