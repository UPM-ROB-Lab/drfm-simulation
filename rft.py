import numpy as np
from scipy.io import loadmat
import pickle
import os
from numba import njit, prange

# 定义常量，避免在 JIT 函数中重复创建
Z_DIR = np.array((0., 0., 1.))

@njit(cache=True)
def numba_trial_func_irt(coeffs, x, mu_f_matrix):
    """
    使用 numba 优化后的 _trial_func_irt 版本。
    参数:
      coeffs: 一维数组，长度至少为 60，对应三个输出多项式的系数。
      x: 二维数组，形状 (N, 3)；每一行对应 [beta, gamma, psi]（原来 _trial_func_irt 中使用的角度）
      mu_f_matrix: 一维数组，形状 (N,)，对应每个点的摩擦系数。
    返回:
      outputArg: 二维数组，形状 (N, 3)，即最终计算得到的输出力。
    """
    N = x.shape[0]
    
    # 计算 x1, x2, x3（原 _trial_func_irt 中的计算逻辑）
    # 注意：此处 x[:,0]=beta, x[:,1]=gamma, x[:,2]=psi
    x1 = np.empty(N)
    x2 = np.empty(N)
    x3 = np.empty(N)
    for i in range(N):
        x1[i] = np.sin(x[i, 1])
        x2[i] = np.cos(x[i, 0])
        x3[i] = np.cos(x[i, 2]) * np.cos(x[i, 1]) * np.sin(x[i, 0]) + np.sin(x[i, 1]) * np.cos(x[i, 0])
    
    # 计算 f1, f2, f3：采用 20 个多项式项
    f1 = np.zeros(N)
    f2 = np.zeros(N)
    f3 = np.zeros(N)
    for term in range(20):
        e1 = term % 3
        e2 = (term // 3) % 3
        e3 = term // 9
        for j in range(N):
            t_val = (x1[j] ** e1) * (x2[j] ** e2) * (x3[j] ** e3)
            f1[j] += coeffs[term] * t_val
            f2[j] += coeffs[20 + term] * t_val
            f3[j] += coeffs[40 + term] * t_val

    # 计算其他角度函数
    sinbeta = np.empty(N)
    cospsi  = np.empty(N)
    cosgamma = np.empty(N)
    sinpsi  = np.empty(N)
    cosbeta = np.empty(N)
    singamma = np.empty(N)
    for i in range(N):
        sinbeta[i] = np.sin(x[i, 0])
        cospsi[i]  = np.cos(x[i, 2])
        cosgamma[i] = np.cos(x[i, 1])
        sinpsi[i]  = np.sin(x[i, 2])
        cosbeta[i] = np.cos(x[i, 0])
        singamma[i] = np.sin(x[i, 1])
    
    fx = np.empty(N)
    fy = np.empty(N)
    fz = np.empty(N)
    for i in range(N):
        fx[i] = f1[i] * sinbeta[i] * cospsi[i] + f2[i] * cosgamma[i]
        fy[i] = f1[i] * sinbeta[i] * sinpsi[i]
        fz[i] = -(f1[i] * cosbeta[i] + f2[i] * singamma[i] + f3[i])
    
    # 计算单位法向量 n = [cos(psi)*sin(beta), sin(psi)*sin(beta), -cos(beta)]
    n = np.empty((N, 3))
    for i in range(N):
        n[i, 0] = np.cos(x[i, 2]) * np.sin(x[i, 0])
        n[i, 1] = np.sin(x[i, 2]) * np.sin(x[i, 0])
        n[i, 2] = -np.cos(x[i, 0])
    
    # sigma_n = (fx, fy, fz) dot n
    sigma_n = np.empty(N)
    for i in range(N):
        sigma_n[i] = fx[i] * n[i, 0] + fy[i] * n[i, 1] + fz[i] * n[i, 2]
    
    # sigma_t = (fx, fy, fz) - sigma_n * n
    sigma_t = np.empty((N, 3))
    for i in range(N):
        sigma_t[i, 0] = fx[i] - sigma_n[i] * n[i, 0]
        sigma_t[i, 1] = fy[i] - sigma_n[i] * n[i, 1]
        sigma_t[i, 2] = fz[i] - sigma_n[i] * n[i, 2]
    
    # sigma_t 的模长
    sigma_t_mag = np.empty(N)
    for i in range(N):
        sigma_t_mag[i] = np.sqrt(sigma_t[i, 0]**2 + sigma_t[i, 1]**2 + sigma_t[i, 2]**2)
    
    sigma_t_mag_new = np.empty(N)
    sigma_t_scaling = np.empty(N)
    for i in range(N):
        abs_val = np.abs(sigma_n[i] * mu_f_matrix[i])
        if sigma_t_mag[i] > abs_val:
            sigma_t_mag_new[i] = abs_val
        else:
            sigma_t_mag_new[i] = sigma_t_mag[i]
        if sigma_t_mag[i] >= 1e-5:
            sigma_t_scaling[i] = sigma_t_mag_new[i] / sigma_t_mag[i]
        else:
            sigma_t_scaling[i] = 0.0

    for i in range(N):
        sigma_t[i, 0] *= sigma_t_scaling[i]
        sigma_t[i, 1] *= sigma_t_scaling[i]
        sigma_t[i, 2] *= sigma_t_scaling[i]
    
    # 最终输出
    outputArg = np.empty((N, 3))
    for i in range(N):
        outputArg[i, 0] = sigma_n[i] * n[i, 0] + sigma_t[i, 0]
        outputArg[i, 1] = sigma_n[i] * n[i, 1] + sigma_t[i, 1]
        outputArg[i, 2] = sigma_n[i] * n[i, 2] + sigma_t[i, 2]
    
    return outputArg

@njit(parallel=False, fastmath=True, cache=True)  # 先关闭并行，避免复杂性
def _find_force_function_new_jit(ref_position_matrix, ref_norm_dir_matrix, area_matrix,
                                global_v, global_omega, global_orientation, global_position,
                                xi, expected_sliding_friction, coeffs):
    """
    JIT 优化版本的 _find_force_function_new 函数。
    使用 numba 优化的部分计算力与力矩，并返回更新后的位置和求和后的力与力矩。
    """
    N = ref_norm_dir_matrix.shape[0]
    new_position_matrix = ref_position_matrix.copy().astype(np.float64)
    new_norm_dir_matrix = ref_norm_dir_matrix.copy().astype(np.float64)
    
    # 确保所有输入都是 float64 类型
    global_v = np.asarray(global_v, dtype=np.float64)
    global_omega = np.asarray(global_omega, dtype=np.float64)
    global_position = np.asarray(global_position, dtype=np.float64)
    area_matrix = np.asarray(area_matrix, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)

    # 计算速度矩阵及其单位方向
    v_matrix = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        cross_product = np.cross(global_omega, new_position_matrix[i])
        v_matrix[i] = global_v + cross_product

    # 手动计算 norm，替代 np.linalg.norm(v_matrix, axis=1, keepdims=True)
    v_dir_matrix_norm = np.zeros(N, dtype=np.float64)
    for i in range(N):
        v_dir_matrix_norm[i] = np.sqrt(v_matrix[i, 0]**2 + v_matrix[i, 1]**2 + v_matrix[i, 2]**2)
    
    v_dir_matrix = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        if v_dir_matrix_norm[i] != 0:
            v_dir_matrix[i] = v_matrix[i] / v_dir_matrix_norm[i]

    tol = 1e-10
    # 直接定义 z_dir_matrix，避免使用全局常量
    z_dir_matrix = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    r_dir_matrix = v_dir_matrix.copy()
    r_dir_matrix[:, 2] = 0.0
    
    # 手动计算 norm
    r_dir_matrix_norm = np.zeros(N, dtype=np.float64)
    for i in range(N):
        r_dir_matrix_norm[i] = np.sqrt(r_dir_matrix[i, 0]**2 + r_dir_matrix[i, 1]**2 + r_dir_matrix[i, 2]**2)
    
    for i in range(N):
        if r_dir_matrix_norm[i] != 0:
            r_dir_matrix[i] = r_dir_matrix[i] / r_dir_matrix_norm[i]
    
    # 处理零向量情况
    for i in range(N):
        if r_dir_matrix_norm[i] < tol:
            r_dir_matrix[i] = new_norm_dir_matrix[i]
    
    r_dir_matrix[:, 2] = 0.0
    
    # 再次计算 norm
    for i in range(N):
        r_dir_matrix_norm[i] = np.sqrt(r_dir_matrix[i, 0]**2 + r_dir_matrix[i, 1]**2 + r_dir_matrix[i, 2]**2)
    
    for i in range(N):
        if r_dir_matrix_norm[i] != 0:
            r_dir_matrix[i] = r_dir_matrix[i] / r_dir_matrix_norm[i]

    # 计算 t_dir_matrix
    t_dir_matrix = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        t_dir_matrix[i] = np.cross(z_dir_matrix, r_dir_matrix[i])
    
    # 手动计算 norm
    t_dir_matrix_norm = np.zeros(N, dtype=np.float64)
    for i in range(N):
        t_dir_matrix_norm[i] = np.sqrt(t_dir_matrix[i, 0]**2 + t_dir_matrix[i, 1]**2 + t_dir_matrix[i, 2]**2)
    
    for i in range(N):
        if t_dir_matrix_norm[i] != 0:
            t_dir_matrix[i] = t_dir_matrix[i] / t_dir_matrix_norm[i]

    # 角度计算
    dot_v_dir_r = np.zeros(N, dtype=np.float64)
    for i in range(N):
        dot_v_dir_r[i] = v_dir_matrix[i, 0] * r_dir_matrix[i, 0] + v_dir_matrix[i, 1] * r_dir_matrix[i, 1] + v_dir_matrix[i, 2] * r_dir_matrix[i, 2]
    
    dot_v_dir_z = v_dir_matrix[:, 2].copy()
    
    # 手动实现 clip
    for i in range(N):
        if dot_v_dir_r[i] > 1.0:
            dot_v_dir_r[i] = 1.0
        elif dot_v_dir_r[i] < -1.0:
            dot_v_dir_r[i] = -1.0
    
    gamma_matrix = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sign_val = 1.0 if -dot_v_dir_z[i] >= 0 else -1.0
        gamma_matrix[i] = np.arccos(dot_v_dir_r[i]) * sign_val

    # 计算法向量相关的点积
    dot_n_r = np.zeros(N, dtype=np.float64)
    dot_n_t = np.zeros(N, dtype=np.float64)
    for i in range(N):
        dot_n_r[i] = new_norm_dir_matrix[i, 0] * r_dir_matrix[i, 0] + new_norm_dir_matrix[i, 1] * r_dir_matrix[i, 1] + new_norm_dir_matrix[i, 2] * r_dir_matrix[i, 2]
        dot_n_t[i] = new_norm_dir_matrix[i, 0] * t_dir_matrix[i, 0] + new_norm_dir_matrix[i, 1] * t_dir_matrix[i, 1] + new_norm_dir_matrix[i, 2] * t_dir_matrix[i, 2]
    
    dot_n_z = new_norm_dir_matrix[:, 2].copy()
    
    # 构建 new_norm_dir_matrix_rtz
    new_norm_dir_matrix_rtz = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        new_norm_dir_matrix_rtz[i, 0] = dot_n_r[i]
        new_norm_dir_matrix_rtz[i, 1] = dot_n_t[i]
        new_norm_dir_matrix_rtz[i, 2] = dot_n_z[i]

    # 反射处理
    for i in range(N):
        reflection = -1.0 if dot_n_r[i] < 0 else 1.0
        new_norm_dir_matrix_rtz[i, 0] *= reflection
        new_norm_dir_matrix_rtz[i, 1] *= reflection
        new_norm_dir_matrix_rtz[i, 2] *= reflection
    
    beta_matrix = np.zeros(N, dtype=np.float64)
    for i in range(N):
        pi_factor = np.pi if new_norm_dir_matrix_rtz[i, 2] < 0 else 0.0
        beta_matrix[i] = -np.arccos(new_norm_dir_matrix_rtz[i, 2]) + pi_factor

    n_rt_matrix = new_norm_dir_matrix_rtz.copy()
    n_rt_matrix[:, 2] = 0.0
    
    # 手动计算 norm
    n_rt_matrix_norm = np.zeros(N, dtype=np.float64)
    for i in range(N):
        n_rt_matrix_norm[i] = np.sqrt(n_rt_matrix[i, 0]**2 + n_rt_matrix[i, 1]**2 + n_rt_matrix[i, 2]**2)
    
    for i in range(N):
        if n_rt_matrix_norm[i] != 0:
            n_rt_matrix[i] = n_rt_matrix[i] / n_rt_matrix_norm[i]
    
    # 处理零向量情况
    for i in range(N):
        if n_rt_matrix_norm[i] < tol:
            n_rt_matrix[i] = r_dir_matrix[i]
    
    # 再次计算 norm
    for i in range(N):
        n_rt_matrix_norm[i] = np.sqrt(n_rt_matrix[i, 0]**2 + n_rt_matrix[i, 1]**2 + n_rt_matrix[i, 2]**2)
    
    for i in range(N):
        if n_rt_matrix_norm[i] != 0:
            n_rt_matrix[i] = n_rt_matrix[i] / n_rt_matrix_norm[i]

    psi_matrix = np.zeros(N, dtype=np.float64)
    for i in range(N):
        psi_val = np.arctan2(n_rt_matrix[i, 1], n_rt_matrix[i, 0])
        psi_matrix[i] = np.abs(psi_val)
        # 手动处理 NaN
        if np.isnan(psi_matrix[i]):
            psi_matrix[i] = 0.0

    sign_fy = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sign_fy[i] = -1.0 if n_rt_matrix[i, 1] < 0 else 1.0

    # 构造传入 numba 加速函数的 x 数组，每行为 [beta, gamma, psi]
    x_input = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        x_input[i, 0] = beta_matrix[i]
        x_input[i, 1] = gamma_matrix[i]
        x_input[i, 2] = psi_matrix[i]
    
    mu_f_array = np.full(N, expected_sliding_friction, dtype=np.float64)
    force_norm_rft = numba_trial_func_irt(coeffs, x_input, mu_f_array)
    
    # 计算 force_matrix，并调整方向
    force_matrix = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        force_matrix[i, 0] = xi * force_norm_rft[i, 0]
        force_matrix[i, 1] = xi * force_norm_rft[i, 1] * sign_fy[i]
        force_matrix[i, 2] = xi * force_norm_rft[i, 2]
        
        # 乘以面积
        force_matrix[i, 0] *= area_matrix[i]
        force_matrix[i, 1] *= area_matrix[i]
        force_matrix[i, 2] *= area_matrix[i]

    # 考虑深度依赖性
    depth = np.zeros(N, dtype=np.float64)
    for i in range(N):
        depth[i] = -(global_position[2] + new_position_matrix[i, 2])
        
        # 检查条件
        H1 = depth[i] < 0
        H2 = (v_dir_matrix[i, 0] * new_norm_dir_matrix[i, 0] + 
              v_dir_matrix[i, 1] * new_norm_dir_matrix[i, 1] + 
              v_dir_matrix[i, 2] * new_norm_dir_matrix[i, 2]) < -0.001
        
        if H1 or H2:
            depth[i] = 0.0
        
        # 应用深度因子
        force_matrix[i, 0] *= depth[i]
        force_matrix[i, 1] *= depth[i]
        force_matrix[i, 2] *= depth[i]

    # 转换到全局坐标：利用 r_dir_matrix, t_dir_matrix, z_dir_matrix
    force_global = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        for j in range(3):
            force_global[i, j] = (force_matrix[i, 0] * r_dir_matrix[i, j] + 
                                 force_matrix[i, 1] * t_dir_matrix[i, j] + 
                                 force_matrix[i, 2] * z_dir_matrix[j])

    # 计算力矩
    torque_matrix = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        torque_matrix[i] = np.cross(new_position_matrix[i], force_global[i])

    # 计算位置 - 修复类型不匹配问题
    position_matrix = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        for j in range(3):
            position_matrix[i, j] = global_position[j] + new_position_matrix[i, j]

    # 计算功率
    power_per_element = np.zeros(N, dtype=np.float64)
    for i in range(N):
        power_per_element[i] = (force_global[i, 0] * v_matrix[i, 0] + 
                               force_global[i, 1] * v_matrix[i, 1] + 
                               force_global[i, 2] * v_matrix[i, 2])
    
    total_element_power = np.sum(power_per_element)

    # 对 force_matrix 和 torque_matrix 求和
    force_total = np.sum(force_global, axis=0)
    torque_total = np.sum(torque_matrix, axis=0)

    # 返回更新后的位置, 总力, 总力矩, 和基于面元计算的总功率
    return position_matrix, force_total, torque_total, total_element_power

class RFTCalculator:
    """
    Responsible for calculating forces acting on a wheel based on RFT principles.
    Singleton class - use get_instance() to get the instance.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, params=None):
        if cls._instance is None:
            if params is None:
                raise ValueError("params must be provided for first initialization")
            cls._instance = cls(params)
        return cls._instance

    def __init__(self, params):
        self.params = params
        self.mesh = None
        self.latest_coeff = None
        self.shape_matrix = None
        # 在初始化时加载形状矩阵和系数文件
        self._load_shape_matrix()
        self._load_coefficients()  # 只调用一次，后续直接使用 self.latest_coeff


    def _load_shape_matrix(self):
        try:
            with open(self.params['left_shape_matrix_path'], 'rb') as f:
                self.left_shape_matrix = pickle.load(f)
            with open(self.params['right_shape_matrix_path'], 'rb') as f:
                self.right_shape_matrix = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading shape matrix: {e}")

    def _load_coefficients(self):
        if not os.path.exists(self.params['coefficients_path']):
            raise FileNotFoundError(f"Coefficients file not found: {self.params['coefficients_path']}")
        self.latest_coeff = loadmat(self.params['coefficients_path'])

    def initialize_mesh(self, wheel_geometry):
        pass

    def compute_forces(self, wheel_data, dt):
        parameters = {
            'medium_internal_friction': self.params.get('internal_friction', 0.4),
            'medium_density': self.params.get('density', 700),
            'gravity': self.params.get('gravity', 9.81),
            'surface_sliding_friction': self.params.get('sliding_friction', 0.2),
            'time_step': dt,
            'position': wheel_data['position'],
            'initial_position': wheel_data['absolute_position'],
            'initial_orientation': wheel_data['phi'],
            'velocity': wheel_data['velocity'],
            'angular_velocity': wheel_data['omega']
        }
        if wheel_data['position'][0] < 0:
            shape_matrix = self.left_shape_matrix
        else:
            shape_matrix = self.right_shape_matrix
            
        # 接收 total_element_power
        new_position, force_total, torque_total, total_element_power = self._run_single_simulation_step(
            shape_matrix, self.latest_coeff, parameters)

        # 返回总力、总力矩和总功率
        return force_total, torque_total, total_element_power

    def _run_single_simulation_step(self, shape_matrix, latest_coeff, parameters):
        mu = parameters['medium_internal_friction']
        rho_c = parameters['medium_density']
        g = parameters['gravity']
        scaling_coeff = self.params['scaling_coeff'] # 从 self.params 读取配置值
        scaleFactor = scaling_coeff * rho_c * g * (894 * mu**3 - 386 * mu**2 + 89 * mu) # 使用配置文件中的值
                
        # 假设 shape_matrix 前 3 列为位置，接下来 3 列为法向量，最后一列为面积信息
        ref_position_matrix = shape_matrix[:, 0:3]
        ref_norm_dir_matrix = shape_matrix[:, 3:6]
        area_matrix = shape_matrix[:, 6]
        
        # 正确解包所有四个返回值 - 调用全局 JIT 函数
        new_position, force_total, torque_total, total_element_power = _find_force_function_new_jit(
            ref_position_matrix, ref_norm_dir_matrix, area_matrix,
            parameters['velocity'], parameters['angular_velocity'],
            parameters['initial_orientation'], parameters['initial_position'],
            scaleFactor, parameters['surface_sliding_friction'], latest_coeff['coeffs'].flatten())
        
        # 返回值现在包含 total_element_power
        return new_position, force_total, torque_total, total_element_power

    def _trial_func_irt_numpy(self, coeffs, x, mu_f_matrix):
        """Calculate the resistive forces using RFT coefficients and angles, NumPy version."""
        # 如果 mu_f_matrix 为标量，则扩展为数组
        if np.isscalar(mu_f_matrix):
            mu_f_matrix = np.full(x.shape[0], mu_f_matrix)

        # 根据原代码，计算 x1, x2, x3（输入 x 的形状应为 (N, 3)）
        x1 = np.sin(x[:, 1])
        x2 = np.cos(x[:, 0])
        x3 = np.cos(x[:, 2]) * np.cos(x[:, 1]) * np.sin(x[:, 0]) + np.sin(x[:, 1]) * np.cos(x[:, 0])

        # 预计算 20 项多项式中每一项的指数组合： (e1, e2, e3) = (i % 3, (i // 3) % 3, i // 9)
        exponents = np.array([(i % 3, (i // 3) % 3, i // 9) for i in range(20)])
        e1 = exponents[:, 0]  # shape (20,)
        e2 = exponents[:, 1]
        e3 = exponents[:, 2]

        # 计算 x1, x2, x3 的幂（利用广播机制，生成形状为 (20, N) 的数组）
        x1_pow = np.power(x1, e1[:, None])
        x2_pow = np.power(x2, e2[:, None])
        x3_pow = np.power(x3, e3[:, None])

        # 计算所有多项式项：每一项为 x1_pow * x2_pow * x3_pow，结果形状为 (20, N)
        poly_terms = x1_pow * x2_pow * x3_pow

        # 利用矩阵点乘，一次性计算 f1, f2, f3
        f1 = np.dot(coeffs[0:20], poly_terms)
        f2 = np.dot(coeffs[20:40], poly_terms)
        f3 = np.dot(coeffs[40:60], poly_terms)

        # 后续部分保持不变，继续计算 fx, fy, fz 等
        sinbeta = np.sin(x[:, 0])
        cospsi = np.cos(x[:, 2])
        cosgamma = np.cos(x[:, 1])
        sinpsi = np.sin(x[:, 2])
        cosbeta = np.cos(x[:, 0])
        singamma = np.sin(x[:, 1])

        fx = f1 * sinbeta * cospsi + f2 * cosgamma
        fy = f1 * sinbeta * sinpsi
        fz = -(f1 * cosbeta + f2 * singamma + f3)

        n = np.column_stack((np.cos(x[:, 2]) * np.sin(x[:, 0]),
                            np.sin(x[:, 2]) * np.sin(x[:, 0]),
                            -np.cos(x[:, 0])))

        sigma_n = np.einsum('ij,ij->i', np.column_stack((fx, fy, fz)), n)
        sigma_t = np.column_stack((fx, fy, fz)) - sigma_n[:, None] * n
        sigma_t_mag = np.linalg.norm(sigma_t, axis=1)

        # 使用向量化操作优化 sigma_t 的缩放
        sigma_t_mag_new = np.where(sigma_t_mag > np.abs(sigma_n * mu_f_matrix),
                                np.abs(sigma_n * mu_f_matrix),
                                sigma_t_mag)
        sigma_t_scaling = np.where(sigma_t_mag >= 1e-5,
                                sigma_t_mag_new / sigma_t_mag,
                                0)
        sigma_t = sigma_t_scaling[:, None] * sigma_t

        outputArg = sigma_n[:, None] * n + sigma_t
        return outputArg


    def _find_force_function_new(self, ref_position_matrix, ref_norm_dir_matrix, area_matrix,
                                global_v, global_omega, global_orientation, global_position,
                                xi, expected_sliding_friction, latest_coeff):
        """
        使用 numba 优化的部分计算力与力矩，并返回更新后的位置和求和后的力与力矩。
        原逻辑保持不变，仅修改返回值部分，使得上层仅接收4个返回值。
        """
        coeffs = latest_coeff['coeffs'].flatten()
        N = ref_norm_dir_matrix.shape[0]
        new_position_matrix = ref_position_matrix
        new_norm_dir_matrix = ref_norm_dir_matrix

        # 计算速度矩阵及其单位方向
        v_matrix = global_v + np.cross(global_omega, new_position_matrix)
        v_dir_matrix_norm = np.linalg.norm(v_matrix, axis=1, keepdims=True)
        v_dir_matrix = np.divide(v_matrix, v_dir_matrix_norm, out=np.zeros_like(v_matrix), where=v_dir_matrix_norm != 0)

        tol = 1e-10
        z_dir_matrix = np.array([0, 0, 1])
        r_dir_matrix = np.copy(v_dir_matrix)
        r_dir_matrix[:, 2] = 0
        r_dir_matrix_norm = np.linalg.norm(r_dir_matrix, axis=1, keepdims=True)
        r_dir_matrix = np.divide(r_dir_matrix, r_dir_matrix_norm, out=np.zeros_like(r_dir_matrix), where=r_dir_matrix_norm != 0)
        zero_indices = r_dir_matrix_norm.flatten() < tol
        r_dir_matrix[zero_indices] = new_norm_dir_matrix[zero_indices]
        r_dir_matrix[:, 2] = 0
        r_dir_matrix_norm = np.linalg.norm(r_dir_matrix, axis=1, keepdims=True)
        r_dir_matrix = np.divide(r_dir_matrix, r_dir_matrix_norm, out=np.zeros_like(r_dir_matrix), where=r_dir_matrix_norm != 0)

        t_dir_matrix = np.cross(z_dir_matrix, r_dir_matrix)
        t_dir_matrix_norm = np.linalg.norm(t_dir_matrix, axis=1, keepdims=True)
        t_dir_matrix = np.divide(t_dir_matrix, t_dir_matrix_norm, out=np.zeros_like(t_dir_matrix), where=t_dir_matrix_norm != 0)

        # 角度计算
        dot_v_dir_r = np.einsum('ij,ij->i', v_dir_matrix, r_dir_matrix)
        dot_v_dir_z = v_dir_matrix[:, 2]
        dot_v_dir_r = np.clip(dot_v_dir_r, -1.0, 1.0)
        gamma_matrix = np.arccos(dot_v_dir_r) * np.sign(-dot_v_dir_z)

        dot_n_r = np.einsum('ij,ij->i', new_norm_dir_matrix, r_dir_matrix)
        dot_n_t = np.einsum('ij,ij->i', new_norm_dir_matrix, t_dir_matrix)
        dot_n_z = new_norm_dir_matrix[:, 2]
        new_norm_dir_matrix_rtz = np.vstack((dot_n_r, dot_n_t, dot_n_z)).T

        reflection_matrix = np.where(dot_n_r < 0, -1, 1)
        new_norm_dir_matrix_rtz *= reflection_matrix[:, np.newaxis]
        beta_matrix = -np.arccos(new_norm_dir_matrix_rtz[:, 2]) + np.pi * (new_norm_dir_matrix_rtz[:, 2] < 0)

        n_rt_matrix = new_norm_dir_matrix_rtz.copy()
        n_rt_matrix[:, 2] = 0
        n_rt_matrix_norm = np.linalg.norm(n_rt_matrix, axis=1, keepdims=True)
        n_rt_matrix = np.divide(n_rt_matrix, n_rt_matrix_norm, out=np.zeros_like(n_rt_matrix), where=n_rt_matrix_norm != 0)
        zero_indices = n_rt_matrix_norm.flatten() < tol
        n_rt_matrix[zero_indices] = r_dir_matrix[zero_indices]
        n_rt_matrix_norm = np.linalg.norm(n_rt_matrix, axis=1, keepdims=True)
        n_rt_matrix = np.divide(n_rt_matrix, n_rt_matrix_norm, out=np.zeros_like(n_rt_matrix), where=n_rt_matrix_norm != 0)

        psi_matrix = np.abs(np.arctan2(n_rt_matrix[:, 1], n_rt_matrix[:, 0]))
        psi_matrix = np.nan_to_num(psi_matrix)

        sign_fy = np.where(n_rt_matrix[:, 1] < 0, -1, 1)

        # 构造传入 numba 加速函数的 x 数组，每行为 [beta, gamma, psi]
        x_input = np.vstack((beta_matrix, gamma_matrix, psi_matrix)).T
        mu_f_array = expected_sliding_friction * np.ones(N)
        force_norm_rft = numba_trial_func_irt(coeffs, x_input, mu_f_array)
        # force_norm_rft = self._trial_func_irt_numpy(coeffs, x_input, mu_f_array)
        # 计算 force_matrix，并调整方向
        force_matrix = xi * np.vstack((
            force_norm_rft[:, 0],
            force_norm_rft[:, 1] * sign_fy,
            force_norm_rft[:, 2]
        )).T
        force_matrix *= area_matrix[:, np.newaxis]

        # 考虑深度依赖性
        depth = -(global_position[2] + new_position_matrix[:, 2])
        H1 = depth < 0
        H2 = np.einsum('ij,ij->i', v_dir_matrix, new_norm_dir_matrix) < -0.001
        depth[H1] = 0
        depth[H2] = 0
        force_matrix = (depth[:, np.newaxis]) * force_matrix

        # 转换到全局坐标：利用 r_dir_matrix, t_dir_matrix, z_dir_matrix
        force_matrix = force_matrix[:, 0][:, np.newaxis] * r_dir_matrix + \
                    force_matrix[:, 1][:, np.newaxis] * t_dir_matrix + \
                    force_matrix[:, 2][:, np.newaxis] * z_dir_matrix

        torque_matrix = np.cross(new_position_matrix, force_matrix)
        position_matrix = global_position + new_position_matrix

        # 新增：计算基于面元力和速度的总功率
        # F_i · v_i for each element, then sum
        # 注意：力 F_i (force_matrix[i]) 是介质施加给面元的力
        # 能量消耗是车轮克服这些力做的功，所以功率是 - Σ (F_i · v_i)
        # 或者等效地，计算 Σ (F_i · v_i)，然后在 simulator 中取负值或绝对值
        # 这里我们计算 P = Σ (F_i · v_i)，表示介质对车轮做功的功率（通常为负）
        power_per_element = np.einsum('ij,ij->i', force_matrix, v_matrix)
        total_element_power = np.sum(power_per_element)

        # 对 force_matrix 和 torque_matrix 求和
        force_total = np.sum(force_matrix, axis=0)
        torque_total = np.sum(torque_matrix, axis=0)
        new_position = position_matrix  # 更新后的位置

        # 返回更新后的位置, 总力, 总力矩, 和基于面元计算的总功率
        return new_position, force_total, torque_total, total_element_power
