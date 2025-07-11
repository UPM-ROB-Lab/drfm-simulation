"""
run_single_simulation.py

作者: [请在此处填写您的姓名]
创建日期: 2025年4月11日
修改日期: 2025年4月15日 (重构以分离计算和保存逻辑)
功能:
    运行车辆转向的单次仿真，并将详细的仿真结果保存到 CSV 和 MAT 文件。
    用户可以通过 JSON 文件或命令行参数修改输入参数。
输出:
    - summary.json: 包含输入参数和总体仿真结果。
    - timeseries_basic.csv: 基本的时间序列数据（位置、速度、航向、能耗）。
    - wheel_details.csv: 详细的车轮力数据。
    - trajectory_actual.csv: 车辆的实际运行轨迹。
    - trajectory_target.csv: 预设的目标轨迹。
    - simulation_data.mat: 包含所有关键仿真数据的 MATLAB .mat 文件 (已压缩)。
    - trajectory_comparison.png: 目标轨迹与实际轨迹的对比图。
"""

import numpy as np
import yaml
import os
import pandas as pd
import scipy.io as sio
import datetime
from simulator import Simulator # 假设 simulator.py 在项目根目录或 Python 路径中
import matplotlib.pyplot as plt # 用于保存轨迹图
import json                     # 用于处理 JSON 文件
import argparse                 # 用于处理命令行参数
import pathlib                  # 用于处理路径对象转换
import plotter                  # 新增：导入 plotter 以调用 Plotter 类

# --- 输出目录 ---
OUTPUT_DIR = "single_result"

# --- 辅助函数：保存仿真结果 ---
def save_simulation_results(run_output_dir, params, sim_dt, sim_total_time, sim_time_steps, results, config_path,
                            timestamps, positions, velocities, headings, wheel_forces, wheel_torques, wheel_positions, energy_per_step, # wheel_torques shape is (n, 4, 3)
                            target_trajectory_arr, plot_timestamp, timing_data=None): # 添加 timing_data 参数
    """
    将仿真结果保存到各种文件格式。

    Args:
        run_output_dir (str): 本次运行的输出目录路径。
        params (dict): 仿真输入参数。
        sim_dt (float): 仿真时间步长。
        sim_total_time (float): 仿真总时间。
        sim_time_steps (int): 仿真总步数。
        results (dict): 仿真运行返回的总体结果。
        config_path (str): 使用的配置文件的绝对路径字符串。
        timestamps (np.ndarray): 时间戳数组。
        positions (np.ndarray): 车辆位置数组 (x, y, z, theta)。
        velocities (np.ndarray): 车辆速度数组 (vx, vy, vz, omega)。
        headings (np.ndarray): 车辆航向角数组 (theta)。
        wheel_forces (np.ndarray): 车轮力数组 (n_steps, 4, 3)。
        wheel_torques (np.ndarray): 车轮力矩数组 (n_steps, 4, 3) -> (Mx, My, Mz)。
        wheel_positions (np.ndarray): 车轮位置数组 (n_steps, 4, 3) -> (x, y, z)。
        energy_per_step (np.ndarray): 每一步的能耗数组。
        target_trajectory_arr (np.ndarray): 目标轨迹点数组 (x, y)。
        plot_timestamp (str): 用于绘图标题的时间戳字符串 (YYYYMMDD_HHMMSS)。
        timing_data (dict, optional): 包含每一步各部分计算时间的字典。 Defaults to None.

    Returns:
        list: 成功生成的文件的路径列表。
    """
    print("保存结果文件...")
    generated_files = []

    # 1. 保存输入参数和总体结果 (JSON)
    summary_data = {
        "input_parameters": params,
        "simulation_config": {
            "dt": sim_dt,
            "total_time": sim_total_time,
            "steps": sim_time_steps,
        },
        "overall_results": {
            "total_energy_consumption": results['total_energy_consumptions'],
            "avg_trajectory_error": results['avg_trajectory_error'],
            "total_distance_travelled": results['total_distance_travelled'],
            "average_energy_per_meter": (results['total_energy_consumptions'] / results['total_distance_travelled']
                                         if results['total_distance_travelled'] > 1e-6 else float('inf')),
        },
        "config_file_used": config_path, # 使用传入的绝对路径字符串
        "output_directory": run_output_dir
    }
    summary_path = os.path.join(run_output_dir, "summary.json")
    try:
        # 确保 numpy 类型、Path 对象、日期时间等可序列化
        def default_serializer(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32,
                                  np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)): # 处理数组
                return obj.tolist() # 将数组转为列表
            elif isinstance(obj, pathlib.Path): # 处理 Path 对象
                return str(obj)
            elif isinstance(obj, (datetime.datetime, datetime.date)): # 处理日期时间
                return obj.isoformat()
            elif isinstance(obj, (bool, np.bool_)): # 处理布尔值
                 return bool(obj)
            elif obj is None: # 处理 None
                 return None # JSON 标准支持 null
            # 添加其他需要处理的类型
            try:
                # 尝试让默认 JSON 编码器处理
                # 如果不行，尝试转换为字符串作为后备
                return str(obj)
            except Exception:
                 raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, default=default_serializer)
        generated_files.append(summary_path)
        print(f"- 已保存: {summary_path}")
    except TypeError as e:
        print(f"错误：序列化 summary_data 到 JSON 时失败: {e}")
    except IOError as e:
        print(f"错误：写入 summary.json 文件失败: {e}")
    except Exception as e:
        print(f"错误：保存 summary.json 时发生未知错误: {e}")


    # 2. 保存时间序列数据 (CSV)
    try:
        timeseries_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Position_X': positions[:, 0],
            'Position_Y': positions[:, 1],
            'Velocity_X': velocities[:, 0],
            'Velocity_Y': velocities[:, 1],
            'Heading': headings,
            'Energy_Consumption_Step': energy_per_step
        })
        timeseries_csv_path = os.path.join(run_output_dir, "timeseries_basic.csv")
        timeseries_df.to_csv(timeseries_csv_path, index=False, float_format='%.6f')
        generated_files.append(timeseries_csv_path)
        print(f"- 已保存: {timeseries_csv_path}")
    except IndexError:
         print("错误：提取位置、速度或能量数据时索引超出范围。检查数据维度。")
    except Exception as e:
        print(f"错误：保存 timeseries_basic.csv 失败: {e}")

    # 3. 保存车轮详细数据 (CSV)
    try:
        wheel_names = ['FL', 'FR', 'RL', 'RR']
        all_wheel_data_dfs = []
        # 创建包含时间戳的基础 DataFrame
        base_df = pd.DataFrame({'Timestamp': timestamps})
        all_wheel_data_dfs.append(base_df)

        for i, name in enumerate(wheel_names):
            # 检查 wheel_forces 和 wheel_torques 维度是否足够
            if wheel_forces.shape[1] > i and wheel_forces.shape[2] >= 3 and wheel_torques.shape[1] > i and wheel_torques.shape[2] >= 3:
                wheel_df = pd.DataFrame({
                    # 'Timestamp': timestamps, # 不再需要，将使用 merge
                    f'Force_{name}_X': wheel_forces[:, i, 0],
                    f'Force_{name}_Y': wheel_forces[:, i, 1],
                    f'Force_{name}_Z': wheel_forces[:, i, 2],
                    f'Torque_{name}_X': wheel_torques[:, i, 0], # 添加力矩 X 列
                    f'Torque_{name}_Y': wheel_torques[:, i, 1], # 添加力矩 Y 列
                    f'Torque_{name}_Z': wheel_torques[:, i, 2], # 添加力矩 Z 列
                })
                # 添加索引以便后续合并（如果需要精确对齐）
                # wheel_df.index = base_df.index
                all_wheel_data_dfs.append(wheel_df)
            else:
                print(f"警告：车轮 {name} 的力或力矩数据在数组中索引无效或不完整，跳过保存。")

        # 使用 concat 按列合并所有 DataFrame
        if len(all_wheel_data_dfs) > 1: # 确保除了时间戳还有其他数据
            combined_wheel_df = pd.concat(all_wheel_data_dfs, axis=1)
            # combined_wheel_df = combined_wheel_df.drop('Timestamp', axis=1, errors='ignore') # 如果有多个 Timestamp 列

            wheel_details_csv_path = os.path.join(run_output_dir, "wheel_details.csv")
            combined_wheel_df.to_csv(wheel_details_csv_path, index=False, float_format='%.6f')
            generated_files.append(wheel_details_csv_path)
            print(f"- 已保存: {wheel_details_csv_path}")
        else:
            print("警告：没有有效的车轮力数据可合并保存到 wheel_details.csv。")
    except IndexError:
        print("错误：提取车轮力数据时索引超出范围。检查 wheel_forces 维度。")
    except Exception as e:
        print(f"错误：保存 wheel_details.csv 失败: {e}")


    # 4. 保存目标轨迹和实际轨迹 (CSV)
    try:
        trajectory_df = pd.DataFrame({
            'Actual_X': positions[:, 0],
            'Actual_Y': positions[:, 1],
        })
        target_trajectory_df = pd.DataFrame(target_trajectory_arr, columns=['Target_X', 'Target_Y'])
        # 保存实际轨迹
        actual_traj_path = os.path.join(run_output_dir, "trajectory_actual.csv")
        trajectory_df.to_csv(actual_traj_path, index=False, float_format='%.6f')
        generated_files.append(actual_traj_path)
        print(f"- 已保存: {actual_traj_path}")
        # 保存目标轨迹
        target_traj_path = os.path.join(run_output_dir, "trajectory_target.csv")
        target_trajectory_df.to_csv(target_traj_path, index=False, float_format='%.6f')
        generated_files.append(target_traj_path)
        print(f"- 已保存: {target_traj_path}")
    except IndexError:
         print("错误：提取轨迹数据时索引超出范围。检查 positions 和 target_trajectory_arr 维度。")
    except Exception as e:
        print(f"错误：保存 trajectory_*.csv 失败: {e}")

    # 4.5. 保存车轮位置数据 (CSV)
    try:
        wheel_names = ['FL', 'FR', 'RL', 'RR']
        # 创建包含时间戳和车轮位置的DataFrame
        wheel_positions_data = {'Timestamp': timestamps}
        for i, name in enumerate(wheel_names):
            if wheel_positions.shape[1] > i and wheel_positions.shape[2] >= 3:
                wheel_positions_data[f'Wheel_{name}_X'] = wheel_positions[:, i, 0]
                wheel_positions_data[f'Wheel_{name}_Y'] = wheel_positions[:, i, 1]
                wheel_positions_data[f'Wheel_{name}_Z'] = wheel_positions[:, i, 2]
            else:
                print(f"警告：车轮 {name} 的位置数据在数组中索引无效，跳过保存。")

        wheel_positions_df = pd.DataFrame(wheel_positions_data)
        wheel_positions_csv_path = os.path.join(run_output_dir, "wheel_positions.csv")
        wheel_positions_df.to_csv(wheel_positions_csv_path, index=False, float_format='%.6f')
        generated_files.append(wheel_positions_csv_path)
        print(f"- 已保存: {wheel_positions_csv_path}")
    except IndexError:
        print("错误：提取车轮位置数据时索引超出范围。检查 wheel_positions 维度。")
    except Exception as e:
        print(f"错误：保存 wheel_positions.csv 失败: {e}")


    # 5. 保存关键数据到 MAT 文件
    try:
        mat_data = {
            'timestamps': timestamps,
            'positions': positions,
            'velocities': velocities,
            'headings': headings,
            'wheel_forces': wheel_forces,
            'wheel_torques': wheel_torques, # 添加力矩数据
            'wheel_positions': wheel_positions, # 添加车轮位置数据
            'energy_consumption_per_step': energy_per_step,
            'target_trajectory': target_trajectory_arr,
            'input_parameters': params, # 将在下面清理
            'summary_results': results, # 将在下面清理
            'average_energy_per_meter': summary_data['overall_results']['average_energy_per_meter']
        }
        # 清理数据类型以适配 savemat
        def sanitize_for_mat(data):
            if isinstance(data, dict):
                return {k: sanitize_for_mat(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [sanitize_for_mat(item) for item in data]
            elif isinstance(data, pathlib.Path):
                return str(data)
            elif isinstance(data, (bool, np.bool_)):
                 return int(data) # MATLAB 通常用 0/1 表示逻辑值
            elif data is None:
                 return '' # 或者 np.nan, MATLAB 可能处理为空矩阵或 NaN
            # 基本类型和 NumPy 数值类型通常可以直接保存
            elif isinstance(data, (str, int, float, np.number, np.ndarray)):
                 return data
            else:
                # 对于无法处理的类型，尝试转换为字符串或返回占位符
                try:
                    print(f"警告: MAT 文件保存时，将类型 {type(data)} 转换为字符串: {str(data)}")
                    return str(data)
                except Exception as e_conv:
                     print(f"错误: 无法为 MAT 文件序列化类型 {type(data)}。错误: {e_conv}")
                     return f"UNSERIALIZABLE_TYPE_{type(data).__name__}"

        # 递归清理 input_parameters 和 summary_results
        mat_data['input_parameters'] = sanitize_for_mat(mat_data['input_parameters'])
        mat_data['summary_results'] = sanitize_for_mat(mat_data['summary_results'])

        mat_path = os.path.join(run_output_dir, "simulation_data.mat")
        sio.savemat(mat_path, mat_data, do_compression=True) # 添加压缩
        generated_files.append(mat_path)
        print(f"- 已保存: {mat_path}")
    except TypeError as e:
        print(f"错误：序列化数据到 MAT 文件时失败: {e}")
    except IOError as e:
        print(f"错误：写入 simulation_data.mat 文件失败: {e}")
    except Exception as e:
        print(f"错误：保存 simulation_data.mat 失败: {e}")

    # 6. 保存详细计时数据 (CSV) - 新增
    if timing_data is not None and isinstance(timing_data, dict) and timing_data:
        try:
            # 检查所有列表长度是否一致
            list_lengths = {key: len(value) for key, value in timing_data.items()}
            first_len = next(iter(list_lengths.values()))
            if not all(length == first_len for length in list_lengths.values()):
                print(f"警告：计时数据中的列表长度不一致: {list_lengths}。将尝试使用最短长度 {first_len} 创建 DataFrame。")
                # 可以选择填充或截断，这里简单地使用原始数据创建，可能导致错误或 NaN
                # 更健壮的方式是确保 simulator.py 中填充正确

            timing_df = pd.DataFrame(timing_data)
            # 添加时间戳列以便关联
            if len(timestamps) == len(timing_df):
                 timing_df.insert(0, 'Timestamp', timestamps)
            else:
                 print(f"警告：时间戳数量 ({len(timestamps)}) 与计时数据步数 ({len(timing_df)}) 不匹配，无法添加时间戳列。")

            timing_csv_path = os.path.join(run_output_dir, "timing_details.csv")
            timing_df.to_csv(timing_csv_path, index=False, float_format='%.9f') # 使用更高精度保存时间
            generated_files.append(timing_csv_path)
            print(f"- 已保存: {timing_csv_path}")
        except ValueError as e:
            print(f"错误：创建计时数据 DataFrame 失败，可能是由于列表长度不匹配: {e}")
        except IOError as e:
            print(f"错误：写入 timing_details.csv 文件失败: {e}")
        except Exception as e:
            print(f"错误：保存 timing_details.csv 时发生未知错误: {e}")
    else:
        print("信息：未提供计时数据或计时数据为空，跳过保存 timing_details.csv。")


    # 7. 保存轨迹对比图
    try:
        plt.figure(figsize=(10, 8))
        plt.plot(target_trajectory_arr[:, 0], target_trajectory_arr[:, 1], 'r--', linewidth=2, label='Target Trajectory')
        plt.plot(positions[:, 0], positions[:, 1], 'g-', linewidth=2, label='Actual Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f"Single Simulation Trajectory Comparison ({plot_timestamp})")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        # 准备文本，确保参数值是字符串
        param_text = '\n'.join([f"{k}: {str(v)}" for k, v in params.items()])
        avg_energy = summary_data['overall_results']['average_energy_per_meter']
        result_text = (f"E_tot: {results['total_energy_consumptions']:.2f} J\n"
                       f"E_avg: {avg_energy:.2f} J/m\n"
                       f"TrajErr: {results['avg_trajectory_error']:.4f} m\n"
                       f"Dist: {results['total_distance_travelled']:.2f} m")
        plt.text(0.02, 0.98, f"Params:\n{param_text}\n\nResults:\n{result_text}",
                 transform=plt.gca().transAxes, fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        plot_path = os.path.join(run_output_dir, "trajectory_comparison.png")
        plt.savefig(plot_path)
        plt.close() # 关闭图形，释放内存
        generated_files.append(plot_path)
        print(f"- 已保存: {plot_path}")
    except IndexError:
         print("错误：绘制轨迹图时数据索引超出范围。")
    except Exception as e:
        print(f"错误：保存 trajectory_comparison.png 失败: {e}")


    print("\n--- 数据保存完成 ---")
    print("生成的文件列表:")
    for file in generated_files:
        print(f"- {file}")
    print("--------------------\n")

    return generated_files


# --- 主执行函数 ---
def run_single_simulation(params, config_path='config.yaml', output_dir=OUTPUT_DIR, progress_callback=None):
    """
    运行单次仿真，提取数据，然后调用保存函数。
    """
    print("开始单次仿真...")

    # --- 加载配置 ---
    print(f"加载配置文件 '{config_path}' ...")
    try:
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_path}' 未找到。")
        return None
    except yaml.YAMLError as e:
        print(f"错误：解析配置文件 '{config_path}' 失败: {e}")
        return None
    print("配置文件加载成功。")

    # --- 创建输出目录 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    try:
        os.makedirs(run_output_dir, exist_ok=True)
    except OSError as e:
        print(f"错误：创建输出目录 '{run_output_dir}' 失败: {e}")
        return None
    print(f"结果将保存在: {run_output_dir}")

    # --- 生成目标轨迹 ---
    print("生成目标轨迹 ...")
    try:
        # ---------------------------------------------------------------
        # 新逻辑：直接使用参数 turning_radius（支持符号表示转向方向）
        # 如果未提供 turning_radius，则回退到旧的 steering_angle 计算逻辑，
        # 以保持向后兼容。
        # ---------------------------------------------------------------

        wheelbase = config.get('vehicle', {}).get('wheelbase', 0.6)  # 仅在需要时使用
        # 获取前轮轮距（track_width），优先 vehicle.track_width，否则自动从 wheels 推算，否则默认0.18
        track_width = config.get('vehicle', {}).get('track_width', None)
        if track_width is None:
            wheels = config.get('wheels', [])
            if len(wheels) >= 2:
                try:
                    y_fl = wheels[0]['position'][1] if isinstance(wheels[0], dict) else wheels[0][1]
                    y_fr = wheels[1]['position'][1] if isinstance(wheels[1], dict) else wheels[1][1]
                    track_width = abs(y_fl - y_fr)
                except Exception:
                    track_width = 0.18
            else:
                track_width = 0.18

        # 1) 首选: 直接读取 turning_radius（允许为负，负值表示左转，正值表示右转）
        if 'turning_radius' in params:
            turning_radius_raw = params['turning_radius']
            if turning_radius_raw == 0:
                raise ValueError("参数 'turning_radius' 不能为 0。")
            turning_radius = abs(turning_radius_raw)
            is_left_turn = turning_radius_raw < 0  # 约定: 负值表示左转 (逆时针)，正值表示右转 (顺时针)
        # 2) 兼容旧参数: 根据 steering_angle 计算
        elif 'steering_angle' in params:
            steering_angle_rad = np.radians(params['steering_angle'])
            # 避免 tan(pi/2) 等情况
            if abs(np.cos(steering_angle_rad)) < 1e-9 or abs(np.tan(steering_angle_rad)) < 1e-9:
                turning_radius = float('inf')
            else:
                turning_radius = wheelbase / np.tan(steering_angle_rad) + track_width / 2
            is_left_turn = params['steering_angle'] > 0  # 原有定义: 正值左转
        else:
            raise KeyError("缺少 'turning_radius' 或 'steering_angle'，无法确定车辆转向半径。")

        # 统一处理无穷大 (直线)
        if turning_radius == float('inf') or np.isinf(turning_radius):
            is_left_turn = False  # 方向无意义

        # 确保 dt 和 total_time 存在且为正
        dt = params.get('dt', 0.01)
        total_time = params.get('total_time', 10.0)
        max_speed = params.get('max_speed', 1.0)
        if dt <= 0 or total_time <= 0:
            raise ValueError("参数 'dt' 和 'total_time' 必须是正数。")

        num_trajectory_points = max(100, int(total_time / dt) * 2) # 保证至少有100个点

        # 新增：支持 arc_fraction 或 arc_angle 参数，允许生成半圆或任意圆弧
        arc_fraction = params.get('arc_fraction', None)  # 0~1, 0.5为半圆，1为整圆
        arc_angle = params.get('arc_angle', None)        # 角度，180为半圆，360为整圆

        circle_center_x, circle_center_y = None, None # Initialize - Corrected indentation
        circle_center = None # Initialize circle_center for straight line case - Corrected indentation
        if abs(turning_radius) == float('inf'): # 直线
            target_trajectory = [(0, y) for y in np.linspace(0, max_speed * total_time, num_trajectory_points)]
            print("生成直线目标轨迹...")
            # For straight lines, circle center and radius are not applicable for error calc
            # turning_radius is already inf
            circle_center = None # Explicitly set to None for straight lines
        else: # 转弯
            # 默认方式：由 total_time 和 max_speed 决定弧长
            arc_length = max_speed * total_time
            theta_end_factor = arc_length / abs(turning_radius)

            # 优先使用 arc_angle 或 arc_fraction 覆盖 theta_end_factor
            if arc_angle is not None:
                # 用户指定角度（单位：度），正值为逆时针，负值为顺时针
                theta_span = np.radians(abs(arc_angle))
            elif arc_fraction is not None:
                # 用户指定圆弧比例（0~1）
                theta_span = 2 * np.pi * abs(arc_fraction)
            else:
                theta_span = theta_end_factor

            if is_left_turn:  # 左转 (逆时针)
                start_angle = 0.5 * np.pi
                end_angle = start_angle - theta_span
                circle_center_x = -turning_radius
                circle_center_y = 0
                print(f"生成左转目标轨迹...（圆弧跨度: {np.degrees(theta_span):.2f}°）")
            else:  # 右转 (顺时针)
                start_angle = 0.5 * np.pi
                end_angle = start_angle + theta_span
                circle_center_x = turning_radius  # 恢复原逻辑，通过坐标翻转处理
                circle_center_y = 0
                print(f"生成右转目标轨迹...（圆弧跨度: {np.degrees(theta_span):.2f}°）")

            theta = np.linspace(start_angle, end_angle, num_trajectory_points)
            # 生成轨迹点，然后对x坐标进行翻转以符合理想圆心在X轴负半轴的要求
            target_trajectory = [(-1 * (circle_center_x + turning_radius * np.cos(t)),
                                  circle_center_y + turning_radius * np.sin(t)) for t in theta]
            # Store the calculated center for passing to Simulator (也需要翻转x坐标)
            circle_center = (-circle_center_x, circle_center_y)

        target_trajectory_arr = np.array(target_trajectory)
        print("目标轨迹生成完成。")
    except KeyError as e:
        print(f"错误：生成目标轨迹时缺少必要的参数: {e}")
        return None
    except ValueError as e:
        print(f"错误：生成目标轨迹时参数值无效: {e}")
        return None

    # --- 初始化仿真器 ---
    print("初始化仿真器...")
    # Pass circle_center and turning_radius to the Simulator
    sim = Simulator(config, dt=dt, total_time=total_time, plot_mode='none',
                    target_trajectory=target_trajectory, # Keep original trajectory for plotting etc.
                    circle_center=circle_center,         # Pass calculated center
                    turning_radius=turning_radius,
                    progress_callback=progress_callback)
    print("仿真器初始化成功。")

    # --- 运行仿真 ---
    print("运行仿真计算...")
    try:
        # 确保所有需要的参数都存在
        required_params = ['max_speed', 'accel_time', 'const_time',
                           'left_steering_angle', 'right_steering_angle', 'beta_FR', 'beta_RL', 'beta_RR']
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise KeyError(f"仿真运行缺少参数: {', '.join(missing_params)}")

        results = sim.run_steering_optimization_simulation(
            max_speed=params['max_speed'],
            accel_time=params['accel_time'],
            const_time=params['const_time'],
            left_steering_angle=params['left_steering_angle'],
            right_steering_angle=params['right_steering_angle'],
            beta_FR=params['beta_FR'],
            beta_RL=params['beta_RL'],
            beta_RR=params['beta_RR'],
            speed_changed=True, # 假设总是变化
            angle_changed=True  # 假设总是变化
        )
        print("仿真计算完成。")
    except KeyError as e:
        print(f"错误：运行仿真时缺少参数: {e}")
        return None

    # --- 提取详细数据 ---
    print("提取仿真数据...")
    try:
        timestamps = np.array(sim.time)
        positions = np.array(sim.positions) # Shape: (n_steps, 4) -> x, y, z, theta
        if positions.shape[1] < 4:
             raise ValueError(f"位置数据维度不足，期望至少4列，实际为 {positions.shape[1]}")
        headings = positions[:, 3] # 提取航向角 (theta)
        velocities = np.array(sim.velocities) # Shape: (n_steps, 4) -> vx, vy, vz, omega
        wheel_forces = np.array(sim.wheel_forces) # Shape: (n_steps, 4, 3)
        wheel_torques = np.array(sim.wheel_torques) # Expected Shape: (n_steps, 4, 3)
        wheel_positions = np.array(sim.wheel_positions) # Shape: (n_steps, 4, 3) -> x, y, z for each wheel
        print(f"DEBUG: Extracted wheel_torques shape: {wheel_torques.shape}") # 添加打印语句
        print(f"DEBUG: Extracted wheel_positions shape: {wheel_positions.shape}") # 添加打印语句
        energy_per_step = np.array(sim.metrics.energy_consumptions) # 从 metrics 获取

        # 验证数据维度是否匹配时间戳
        n_steps = len(timestamps)
        # 确保所有数组的第一维都等于 n_steps
        data_arrays = [positions, velocities, wheel_forces, wheel_torques, wheel_positions, energy_per_step]
        mismatched_arrays = [i for i, arr in enumerate(data_arrays) if arr.shape[0] != n_steps]

        if mismatched_arrays:
             print(f"警告：提取的部分仿真数据步数 ({[data_arrays[i].shape[0] for i in mismatched_arrays]}) 与时间戳数量 ({n_steps}) 不匹配！")
             # 可以在这里添加更复杂的处理逻辑，例如截断所有数组到最短长度
             min_steps = min(arr.shape[0] for arr in data_arrays + [timestamps])
             if min_steps < n_steps:
                 print(f"将所有数据截断到最短长度: {min_steps}")
                 timestamps = timestamps[:min_steps]
                 positions = positions[:min_steps]
                 velocities = velocities[:min_steps]
                 wheel_forces = wheel_forces[:min_steps]
                 wheel_torques = wheel_torques[:min_steps]
                 wheel_positions = wheel_positions[:min_steps]
                 energy_per_step = energy_per_step[:min_steps]
                 n_steps = min_steps # 更新 n_steps

        print("仿真数据提取完成。")
    except AttributeError as e:
         print(f"错误：从 Simulator 对象提取数据时属性不存在: {e}。请检查 Simulator 实现。")
         return None
    except ValueError as e:
         print(f"错误：提取的数据维度不正确: {e}")
         return None

    # --- 保存结果 ---
    print("调用保存函数...")
    # 获取配置文件的绝对路径字符串
    abs_config_path = str(pathlib.Path(config_path).resolve())
    # 从 results 字典中提取 timing_data
    timing_data = results.get('timing_data', None)

    generated_files = save_simulation_results(
        run_output_dir=run_output_dir,
        params=params,
        sim_dt=sim.dt,
        sim_total_time=sim.total_time,
        sim_time_steps=sim.time_steps,
        results=results,
        config_path=abs_config_path, # 传递绝对路径字符串
        timestamps=timestamps,
        positions=positions,
        velocities=velocities,
        headings=headings,
        wheel_forces=wheel_forces,
        wheel_torques=wheel_torques, # 传递力矩数据
        wheel_positions=wheel_positions, # 传递车轮位置数据
        energy_per_step=energy_per_step,
        target_trajectory_arr=target_trajectory_arr,
        plot_timestamp=timestamp, # 使用之前生成的 timestamp
        timing_data=timing_data # 传递计时数据
    )

    # 新增：调用 Plotter 画图，传递 simulator 对象和时间戳
    plot_paths = plotter.Plotter().plot_simulation_results(
        simulator=sim,
        timestamps=timestamps, # 传递可能被截断的时间戳
        plot_mode='save',
        output_dir=run_output_dir
    )
    # Add generated plot paths to the list
    if isinstance(plot_paths, dict):
        for key, path in plot_paths.items():
             if path: # Ensure path is not None or empty
                 generated_files.append(str(path))

    print("单次仿真结束。")
    return generated_files # 返回生成的文件列表

# --- 程序入口 ---
if __name__ == "__main__":
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="运行单次车辆仿真并从 JSON 文件加载参数。")
    parser.add_argument(
        '--params_file',
        type=pathlib.Path, # 使用 pathlib 处理路径
        default=pathlib.Path('run/simulation_params.json'), # 默认参数文件路径
        help='包含仿真参数的 JSON 文件路径。'
    )
    parser.add_argument(
        '--config_file',
        type=pathlib.Path, # 使用 pathlib 处理路径
        default=pathlib.Path('config.yaml'), # 默认配置文件路径
        help='车辆和仿真器配置 YAML 文件路径。'
    )
    parser.add_argument(
        '--output_dir',
        type=pathlib.Path, # 使用 pathlib 处理路径
        default=pathlib.Path(OUTPUT_DIR), # 使用脚本中定义的默认输出目录
        help='保存仿真结果的根目录。'
    )
    args = parser.parse_args()

    # --- 检查文件是否存在 ---
    if not args.params_file.is_file():
        print(f"错误：参数文件 '{args.params_file}' 未找到或不是文件。")
        exit(1)
    if not args.config_file.is_file():
        print(f"错误：配置文件 '{args.config_file}' 未找到或不是文件。")
        exit(1)

    # --- 加载仿真参数 ---
    print(f"从 '{args.params_file}' 加载仿真参数...")
    try:
        with open(args.params_file, 'r', encoding='utf-8') as f:
            simulation_params = json.load(f)
        print("仿真参数加载成功。")
    except json.JSONDecodeError as e:
        print(f"错误：解析参数文件 '{args.params_file}' 失败: {e}")
        exit(1)
    except IOError as e:
         print(f"错误：读取参数文件 '{args.params_file}' 失败: {e}")
         exit(1)

    # --- （可选）验证或计算参数 ---
    if 'total_time' in simulation_params and 'accel_time' in simulation_params:
        total_t = simulation_params['total_time']
        accel_t = simulation_params['accel_time']
        if total_t < accel_t:
             print(f"警告：'total_time' ({total_t}) 小于 'accel_time' ({accel_t})。请检查参数。")
             # 可以选择退出或继续，这里选择继续但 const_time 会是负数或零

        expected_const_time = total_t - accel_t
        if 'const_time' not in simulation_params:
            print(f"信息：'const_time' 未在参数文件中提供，将根据 'total_time' 和 'accel_time' 计算: {expected_const_time}")
            simulation_params['const_time'] = expected_const_time
        elif not np.isclose(simulation_params['const_time'], expected_const_time):
             print(f"警告：参数文件中的 'const_time' ({simulation_params['const_time']}) 与 'total_time' - 'accel_time' ({expected_const_time}) 不符。将使用文件中的值。")
             # 可以选择强制使用计算值： simulation_params['const_time'] = expected_const_time

    # --- 检查 simulator 是否能导入 ---
    # (导入在顶部，这里仅作形式检查)
    try:
        _ = Simulator # 检查变量是否存在
    except NameError:
         print("错误：'Simulator' 类未定义。请确保 simulator.py 可访问且无导入错误。")
         exit(1)
    # ImportError 会在脚本开始时被捕获

    # --- 运行仿真 ---
    print(f"\n使用参数文件 '{args.params_file}' 和配置文件 '{args.config_file}' 运行仿真...")
    # 将 Path 对象转换为字符串传递给函数
    generated_files_list = run_single_simulation(
        params=simulation_params,
        config_path=str(args.config_file),
        output_dir=str(args.output_dir)
    )

    # --- 检查结果 ---
    if generated_files_list is None:
        print("\n仿真运行或保存过程中遇到错误，未能成功完成。")
        exit(1) # 以非零状态码退出表示失败
    elif not generated_files_list:
        print("\n仿真运行完成，但似乎没有生成任何文件（可能所有保存步骤都失败了）。")
        # 根据需要决定是否算作失败
        exit(1)
    else:
        print("\n仿真成功完成。")
        # exit(0) # 可选：以零状态码明确表示成功