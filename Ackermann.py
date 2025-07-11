import math

class Ackermann:
    def __init__(self, track_width, wheelbase):
        """
        初始化Ackermann计算器
        :param track_width: 轮距 (m)
        :param wheelbase: 轴距 (m)
        """
        self.track_width = track_width
        self.wheelbase = wheelbase

    def calculate_steering_angles(self, inner_steering_angle_deg, ratio=1.0):
        """
        计算左前轮和右前轮的转向角度
        :param inner_steering_angle_deg: 内侧轮的转向角度 (度)，左转时为正（左前轮），右转时为负（右前轮）
        :param ratio: 转向角度调整比例
        :return: 左前轮和右前轮的转向角度 (弧度)
        """
        if inner_steering_angle_deg == 0:
            return 0.0, 0.0

        delta_i_rad = math.radians(abs(inner_steering_angle_deg))
        is_left_turn = inner_steering_angle_deg > 0
        
        # Use atan2 to avoid division by zero
        delta_o_rad = math.atan2(1, 1/math.tan(delta_i_rad) + ratio * (self.track_width/self.wheelbase))
        
        if delta_o_rad <= 0:  # 避免 delta_o_rad 为负或零，这在物理上可能无意义
            return None, None  # 或者抛出异常

        # Use ternary expressions and unified negative sign assignment
        left_steer_rad = -delta_i_rad if is_left_turn else -delta_o_rad
        right_steer_rad = -delta_o_rad if is_left_turn else -delta_i_rad
        
        return left_steer_rad, right_steer_rad

    def calculate_wheel_speeds(self, inner_steering_angle_deg, max_speed):
        """
        计算每个轮子的速度
        :param inner_steering_angle_deg: 内侧轮的转向角度 (度)，左转时为正（左前轮），右转时为负（右前轮）
        :param max_speed: 车辆最大速度 (m/s)
        :return: 前左、前右、后左、后右轮的速度 (m/s)
        """
        if inner_steering_angle_deg == 0:
            return max_speed, max_speed, max_speed, max_speed

        is_left_turn = inner_steering_angle_deg > 0
        delta_inner_rad = math.radians(abs(inner_steering_angle_deg))

        # 计算内侧前轮的旋转半径
        turn_radius_front_inner = self.wheelbase / math.sin(delta_inner_rad)

        # 计算旋转中心的横向偏移量 (相对于后轴中心)
        center_offset_y = turn_radius_front_inner * math.cos(delta_inner_rad)

        # 计算各个轮子的旋转半径
        
        # 左转：左前轮为内侧
        R_fl = turn_radius_front_inner
        R_fr = math.sqrt((center_offset_y + self.track_width)**2 + self.wheelbase**2)
        R_rl = center_offset_y
        R_rr = abs(center_offset_y + self.track_width)
        

        # 确保旋转半径不为零，避免除零错误
        if R_fl == 0 or R_fr == 0 or R_rl == 0 or R_rr == 0:
            return 0.0, 0.0, 0.0, 0.0

        # 计算轮速
        v_inner = max_speed  # 内侧前轮速度为最大速度

        v_fl = v_inner
        v_fr = v_inner * (R_fr / R_fl)
        v_rl = v_inner * (R_rl / R_fl)
        v_rr = v_inner * (R_rr / R_fl)
        
        if not is_left_turn:
            v_fl, v_fr = v_fr, v_fl
            v_rl, v_rr = v_rr, v_rl

        return v_fl, v_fr, v_rl, v_rr