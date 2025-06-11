"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho) # 热扩散系数
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1 # 空间格点数
Nt = 2000     # 时间步数

def basic_heat_diffusion():
    """
    任务1: 基本热传导模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    u = np.zeros((Nx, Nt))
    u[1:-1, 0] = 100  # 初始条件：内部点100K
    
    r = D * dt / dx**2
    print(f"稳定性参数 r = {r:.4f}")
    
    for j in range(Nt-1):
        u[0, j+1] = 0
        u[-1, j+1] = 0
        for i in range(1, Nx-1):
            u[i, j+1] = u[i, j] + r * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
    
    return u
    
def analytical_solution(n_terms=100):
    """
    任务2: 解析解函数
    
    参数:
        n_terms (int): 傅里叶级数项数
    
    返回:
        np.ndarray: 解析解温度分布
    """
    x = np.linspace(0, L, Nx)
    t = np.arange(Nt) * dt
    u_analytical = np.zeros((Nx, Nt))
    
    for n in range(1, 2*n_terms, 2):
        kn = n * np.pi / L
        amplitude = 400 / (n * np.pi)
        
        spatial = np.sin(kn * x)
        temporal = np.exp(-D * kn**2 * t)
        
        for j in range(Nt):
            u_analytical[:, j] += amplitude * spatial * temporal[j]
    
    return u_analytical
    
def stability_analysis():
    """
    任务3: 数值解稳定性分析
    """
    test_dts = [0.25, 0.5, 0.6]
    results = []
    
    for dt_test in test_dts:
        r = D * dt_test / dx**2
        Nt_test = int(1000 / dt_test)
        
        u = np.zeros((Nx, Nt_test))
        u[1:-1, 0] = 100
        
        for j in range(Nt_test-1):
            u[0, j+1] = 0
            u[-1, j+1] = 0
            for i in range(1, Nx-1):
                u[i, j+1] = u[i, j] + r * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
        
        results.append((r, u))
    
    return results

def different_initial_condition():
    """
    任务4: 不同初始条件模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    u = np.zeros((Nx, Nt))
    
    half_idx = int(0.5 / dx)
    u[1:half_idx, 0] = 100
    u[half_idx:-1, 0] = 50
    
    r = D * dt / dx**2
    
    for j in range(Nt-1):
        u[0, j+1] = 0
        u[-1, j+1] = 0
        for i in range(1, Nx-1):
            u[i, j+1] = u[i, j] + r * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
    
    return u
    
def heat_diffusion_with_cooling():
    """
    任务5: 包含牛顿冷却定律的热传导
    """
    u = np.zeros((Nx, Nt))
    u[1:-1, 0] = 100
    
    r = D * dt / dx**2
    
    for j in range(Nt-1):
        u[0, j+1] = 0
        u[-1, j+1] = 0
        for i in range(1, Nx-1):
            u[i, j+1] = (1 - 2*r - h*dt)*u[i, j] + r*(u[i+1, j] + u[i-1, j])
    
    return u

def plot_3d_solution(u, dx, dt, Nt, title):
    """
    绘制3D温度分布图
    
    参数:
        u (np.ndarray): 温度分布数组
        dx (float): 空间步长
        dt (float): 时间步长
        Nt (int): 时间步数
        title (str): 图表标题
    
    返回:
        None
    
    示例:
        >>> u = np.zeros((100, 200))
        >>> plot_3d_solution(u, 0.01, 0.5, 200, "示例")
    """
    x = np.linspace(0, L, u.shape[0])
    t = np.linspace(0, Nt*dt, u.shape[1])
    X, T = np.meshgrid(x, t)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, u.T, cmap='viridis', rstride=10, cstride=2)
    
    ax.set_xlabel('Position x (m)', fontsize=12)
    ax.set_ylabel('Time t (s)', fontsize=12)
    ax.set_zlabel('Temperature T (K)', fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Temperature (K)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    """
    主函数 - 演示和测试各任务功能
    
    执行顺序:
    1. 基本热传导模拟
    2. 解析解计算
    3. 数值解稳定性分析
    4. 不同初始条件模拟
    5. 包含冷却效应的热传导
    
    注意:
        学生需要先实现各任务函数才能正常运行
    """
    print("=== 铝棒热传导问题学生实现 ===")
    
    # 任务1: 基本热传导模拟
    print("\n>>> 运行任务1: 基本热传导模拟...")
    u_basic = basic_heat_diffusion()
    plot_3d_solution(u_basic, dx, dt, Nt, "Basic Heat Diffusion Simulation")
    
    # 任务2: 解析解
    print("\n>>> 运行任务2: 解析解计算...")
    u_analytical = analytical_solution(n_terms=50)
    plot_3d_solution(u_analytical, dx, dt, Nt, "Analytical Solution (50 Fourier Terms)")
    
    # 任务3: 稳定性分析
    print("\n>>> 运行任务3: 数值解稳定性分析...")
    stability_results = stability_analysis()
    for r, u in stability_results:
        plot_3d_solution(u, dx, dt, u.shape[1], f"Stability Analysis (r={r:.4f})")
    
    # 任务4: 不同初始条件
    print("\n>>> 运行任务4: 不同初始条件模拟...")
    u_diff_init = different_initial_condition()
    plot_3d_solution(u_diff_init, dx, dt, Nt, "Different Initial Conditions Simulation")
    
    # 任务5: 包含冷却效应
    print("\n>>> 运行任务5: 包含牛顿冷却定律的热传导...")
    for h in [0.01, 0.1]:
        u_cooling = heat_diffusion_with_cooling(h=h)
        plot_3d_solution(u_cooling, dx, dt, Nt, f"Heat Diffusion with Cooling (h={h} s⁻¹)")
