import numpy as np
from scipy.integrate import solve_ivp


def coupled_duffing(t, y, delta, alpha, beta, adj_matrix, c=0):
    """
    Defines the coupled Duffing oscillator system of equations.

    Parameters:
    t : float
        Time variable.
    y : array_like
        State vector [x1, v1, x2, v2, ... , xn, vn].
    delta : float
        Damping coefficient.
    alpha : float
        Linear stiffness coefficient.
    beta : float
        Nonlinear stiffness coefficient.
    gamma : float
        Amplitude of the driving force.
    omega : float
        Frequency of the driving force.

    Returns:
    dydt : array_like
        Derivatives [dx1/dt, dv1/dt, dx2/dt, dv2/dt].
    """
    dy = np.zeros_like(y)
    system_num = y.shape[0] // 2
    for i in range(system_num):
        x = y[2 * i]
        v = y[2 * i + 1]
        dxdt = v
        dvdt = -delta[i]*v - alpha[i]*x - beta[i]*x**3
        if system_num < 3:
            # Use the simple chain coupling when only a small number of oscillators is present.
            if i == 0:
                dvdt += c * (y[2] - x)
            elif i == system_num - 1:
                dvdt += c * (y[2 * (system_num - 2)] - x)
            else:
                dvdt += c * (y[2 * (i - 1)] + y[2 * (i + 1)] - 2 * x)
            dy[2 * i] = dxdt
            dy[2 * i + 1] = dvdt
        else:
            # For larger systems, use the supplied adjacency matrix to accumulate coupling terms.
            for j in range(system_num):
                if i != j:  # Avoid self-coupling
                    dvdt += adj_matrix[i, j] * ((y[2 * j] - x)**3)
            dy[2 * i] = dxdt
            dy[2 * i + 1] = dvdt
            
    return dy

def simulate_coupled_duffing(delta, alpha, beta, adj_matrix, y0, t_span, t_eval):
    """
    Simulates the coupled Duffing oscillator system.

    Parameters:
    delta : array_like
        Damping coefficients for each oscillator.
    alpha : array_like
        Linear stiffness coefficients for each oscillator.
    beta : array_like
        Nonlinear stiffness coefficients for each oscillator.
    gamma : array_like
        Amplitudes of the driving forces for each oscillator.
    omega : float
        Frequency of the driving forces.
    y0 : array_like
        Initial state vector [x1(0), v1(0), x2(0), v2(0), ... , xn(0), vn(0)].
    t_span : tuple
        Time span for the simulation (t_start, t_end).
    t_eval : array_like
        Time points at which to store the computed solution.

    Returns:
    result : OdeResult
        The result object returned by solve_ivp containing the simulation results.
    """
    result = solve_ivp(coupled_duffing, t_span, y0, args=(delta, alpha, beta, adj_matrix), t_eval=t_eval)
    return result

if __name__ == "__main__":
    N = 3
    sampling = 5001
    seed = 0

    # Build the nearest-neighbor coupling graph for the coupled system.
    adj_matrix = np.diag(np.ones(N - 1), k=1) + np.diag(np.ones(N - 1), k=-1)

    rng = np.random.default_rng(seed=seed)

    # Sample one set of Duffing parameters and an initial condition for the training trajectory.
    delta = np.round(rng.uniform(0.1, 0.3, N), 2)
    alpha = np.round(rng.uniform(-1.0, -0.5, N), 2)
    beta = np.round(rng.uniform(0.5, 1.0, N), 2)
    y0 = rng.uniform(-1.5, 1.5, 2 * N)

    dt = 0.01

    t_span = (0, (sampling - 1) * dt)
    t_eval = np.linspace(t_span[0], t_span[1], sampling)
    result = simulate_coupled_duffing(delta, alpha, beta, adj_matrix, y0, t_span, t_eval)

    test_data_num = 100
    for i in range(test_data_num):
        # Generate test trajectories by perturbing the nominal initial condition.
        rng_test = np.random.default_rng(i + 31471)
        y0_test_noise = rng_test.uniform(-0.3, 0.3, 2 * N)
        y0_test = y0 + y0_test_noise
        test = simulate_coupled_duffing(
            delta,
            alpha,
            beta,
            adj_matrix,
            y0_test,
            (0, 1001 * dt),
            np.arange(0, 1001 * dt, dt),
        )

    print("Simulation completed.")

