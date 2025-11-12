import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# -----------------------------
# Utilities
# -----------------------------


def gradient_fd(f, dx):
    grad = np.zeros_like(f)
    grad[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
    grad[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dx)
    grad[-1] = (3.0 * f[-1] - 4.0 * f[-2]) / (2.0 * dx)
    return grad


def build_u_center_and_u_face(rho, dx, D, rho_L, rho_R):
    """
      - cell-center 速度: u_center = -D * d/dx log(rho)
      - face 速度:   u_face, 包含邊界面 0 與 N
    這裡用 ghost cell:
      rho_ext[0]   = rho_L
      rho_ext[-1]  = rho_R
      rho_ext[1:-1]= rho
    """
    nx = rho.size
    rho_ext = np.empty(nx + 2)
    rho_ext[1:-1] = rho
    # ghost cells
    rho_ext[0] = rho_L
    rho_ext[-1] = rho_R
    # rho_ext[0] = 2.0 * rho_L - rho_ext[1]
    # rho_ext[-1] = 2.0 * rho_R - rho_ext[-2]

    log_rho_ext = np.log(np.clip(rho_ext, 1e-12, None))

    # cell-center velocity
    u_center = np.zeros(nx)
    u_center[:] = -D * (log_rho_ext[2:] - log_rho_ext[:-2]) / (2.0 * dx)

    # face velocity
    u_face = np.zeros(nx + 1)
    u_face[1:-1] = 0.5 * (u_center[:-1] + u_center[1:])
    u_face[0] = -D * 2.0 * (log_rho_ext[1] - log_rho_ext[0]) / dx
    u_face[-1] = -D * 2.0 * (log_rho_ext[-1] - log_rho_ext[-2]) / dx

    return u_center, u_face


def upwind_flux(q_L, q_R, u_face):
    return u_face * (q_L if u_face > 0.0 else q_R)

# -----------------------------
# FVM Solvers (Upwind)
# -----------------------------


def solve_fvm_I(I_n, u_face, dx, dt):
    """
    I_t + (I u)_x = 0 的一次顯式上風 FVM。
    僅更新 interior cell (1..nx-2)，邊界 cell 先保持，
    真正的 Dirichlet 由 rho 的 ghost cell 施加。
    """
    nx = len(I_n)
    I_next = np.copy(I_n)
    flux = np.zeros(nx + 1)

    # faces j = 1..nx-1
    for j in range(1, nx):
        flux[j] = upwind_flux(I_n[j-1], I_n[j], u_face[j])

    I_next[1:-1] = I_n[1:-1] - (dt / dx) * (flux[2:-1] - flux[1:-2])
    return I_next


def solve_fvm_Y(Y_n, u_face, grad_u, dx, dt):
    """
    Y_t + (Y u)_x = Y * u_x
    """
    nx = len(Y_n)
    Y_next = np.copy(Y_n)
    flux = np.zeros(nx + 1)

    for j in range(1, nx):
        flux[j] = upwind_flux(Y_n[j-1], Y_n[j], u_face[j])

    advection = - (dt / dx) * (flux[2:-1] - flux[1:-2])
    source = dt * (Y_n * grad_u)[1:-1]

    Y_next[1:-1] = Y_n[1:-1] + advection + source
    return Y_next


# -----------------------------
# Main Simulation
# -----------------------------
def run_simulation(D=1.0, X_MIN=0.0, X_MAX=4.0,
                   T_MAX=1.0, NX=201, NT=2501):
    # FVM: NX cells, cell-centered grid
    DX = (X_MAX - X_MIN) / NX
    DT = T_MAX / (NT - 1)
    x = X_MIN + (np.arange(NX) + 0.5) * DX  # cell-centers

    # ===== initial condition =====
    rho0 = 2.0 + np.sin(x)
    rho0 = np.where(rho0 <= 0.0, 1e-12, rho0)

    # Dirichlet BCs
    rho_L = 2.0 + np.sin(X_MIN)
    rho_R = 2.0 + np.sin(X_MAX)

    rho_current = rho0.copy()
    Y_current = x.copy()
    I_current = np.ones_like(rho_current)

    rho_history = [rho_current.copy()]

    for n in range(1, NT):
        # 建立 u_center, u_face
        u_center, u_face = build_u_center_and_u_face(
            rho_current, DX, D, rho_L, rho_R
        )

        # 固定時間步長 → 用 CFL clip 速度
        max_vel = DX / DT
        u_center = np.clip(u_center, -max_vel, max_vel)
        u_face = np.clip(u_face, -max_vel, max_vel)

        # du/dx
        grad_u = gradient_fd(u_center, DX)

        # FVM 更新 I, Y
        I_next = solve_fvm_I(I_current, u_face, DX, DT)
        Y_next = solve_fvm_Y(Y_current, u_face, grad_u, DX, DT)

        Y_clipped = np.clip(Y_next, -np.pi, np.pi)
        rho0_Y = 2.0 + np.sin(Y_clipped)

        # 重建 rho
        rho_next = rho0_Y * I_next
        rho_next = np.clip(rho_next, 1e-12, None)

        # 注意：這裡不再強制 rho_next[0/-1] = rho_L/R
        # Dirichlet 已經透過 ghost cell → u_face → 通量 實現

        rho_history.append(rho_next.copy())
        rho_current = rho_next
        Y_current = Y_next
        I_current = I_next

    return x, np.array(rho_history)

# -----------------------------
# Ground Truth (Exact Solution)
# -----------------------------


def generate_groundtruth(x, T_MAX=1.0, NT=2501, D=1.0):
    """
    解析解：rho(x,t) = 2 + sin(x) * exp(-t)
    滿足 rho_t = D * rho_xx, Dirichlet：rho=2 在邊界。
    這裡直接在 cell-center x 上取樣。
    """
    time_steps = np.linspace(0.0, T_MAX, NT)
    rho_exact_history = []
    for t in time_steps:
        rho_exact = 2.0 + np.sin(x) * np.exp(-t)
        rho_exact_history.append(rho_exact)
    return np.array(rho_exact_history), time_steps

# -----------------------------
# Plotting
# -----------------------------


def plot_results(x, rho_numerical_history, rho_exact_history,
                 time_steps, X_MIN=0.0, X_MAX=4.0,
                 NT=2501, output_dir="output"):
    print("Creating animation...")
    fig, ax = plt.subplots(figsize=(10, 6))
    line_num, = ax.plot([], [], "r--", label="Numerical (FVM Upwind)")
    line_ex, = ax.plot([], [], "g:", lw=3, label="Exact")
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0.5, 3.5)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho(x,t)$")
    ax.set_title("1D Diffusion Equation: Numerical (FVM) vs. Exact Solution")
    ax.legend()
    ax.grid(True)

    sq_err = (rho_numerical_history - rho_exact_history) ** 2
    mse_history = np.mean(sq_err, axis=1)
    max_mse = np.max(mse_history)
    print(f"Maximum MSE across all time points: {max_mse:.6e}")

    max_point_err = np.max(sq_err)
    print("Creating per-point MSE animation...")
    fig_err, ax_err = plt.subplots(figsize=(10, 6))
    line_err, = ax_err.plot([], [], "b-", label="Squared Error")
    err_text = ax_err.text(0.05, 0.9, "", transform=ax_err.transAxes)
    ax_err.set_xlim(X_MIN, X_MAX)
    ax_err.set_ylim(0.0, max_point_err * 1.1 if max_point_err > 0 else 1.0)
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("Squared Error")
    ax_err.set_title("Pointwise Squared Error Over Time")
    ax_err.legend()
    ax_err.grid(True)

    frame_indices = list(range(0, NT, 50))
    if frame_indices[-1] != NT - 1:
        frame_indices.append(NT - 1)

    def update(k):
        n = frame_indices[k]
        rho_num = rho_numerical_history[n]
        rho_ex = rho_exact_history[n]
        mse = np.mean((rho_num - rho_ex) ** 2)
        line_num.set_data(x, rho_num)
        line_ex.set_data(x, rho_ex)
        time_text.set_text(f"t = {time_steps[n]:.2f} s\nMSE = {mse:.2e}")
        return line_num, line_ex, time_text

    def update_err(k):
        n = frame_indices[k]
        se = sq_err[n]
        mse = mse_history[n]
        line_err.set_data(x, se)
        err_text.set_text(f"t = {time_steps[n]:.2f} s\nMSE = {mse:.2e}")
        return line_err, err_text

    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                         interval=40, blit=True)
    writer = PillowWriter(fps=25)
    anim.save(os.path.join(output_dir, "simulation.gif"), writer=writer)
    print(f"Animation saved to {output_dir}/simulation.gif")

    anim_err = FuncAnimation(fig_err, update_err, frames=len(frame_indices),
                             interval=40, blit=True)
    anim_err.save(os.path.join(
        output_dir, "mse.gif"), writer=writer)
    print(f"MSE animation saved to {output_dir}/mse.gif")

# -----------------------------
# Main
# -----------------------------


if __name__ == "__main__":
    output_dir = "results_YI_dirichlet"
    os.makedirs(output_dir, exist_ok=True)

    X_MIN = -np.pi
    X_MAX = np.pi
    T_MAX = 1.0
    NX = 201
    NT = 2501
    D = 1.0

    x, rho_num_hist = run_simulation(D, X_MIN, X_MAX, T_MAX, NX, NT)
    rho_ex_hist, t_steps = generate_groundtruth(x, T_MAX, NT, D)

    plot_results(x, rho_num_hist, rho_ex_hist,
                 t_steps, X_MIN, X_MAX, NT, output_dir=output_dir)
