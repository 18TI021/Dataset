import numpy as np
from scipy.integrate import solve_ivp


def coupled_van_der_pol(t, y, mu, adj_matrix, c=0):
    """
    Defines the coupled Van der Pol oscillator system of equations.

    Parameters:
    t : float
        Time variable.
    y : array_like
        State vector [x1, v1, x2, v2, ... , xn, vn].
    mu : array_like
        Nonlinearity parameters for each oscillator.
    adj_matrix : array_like
        Adjacency matrix describing the coupling between oscillators.
    c : float, optional
        Coupling coefficient used for the nearest-neighbor case when the number
        of oscillators is less than 3.

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
        dvdt = mu[i] * (1 - x**2) * v - x
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
                if i != j:
                    dvdt += adj_matrix[i, j] * (y[2 * j] - x)
            dy[2 * i] = dxdt
            dy[2 * i + 1] = dvdt

    return dy


def simulate_coupled_van_der_pol(mu, adj_matrix, y0, t_span, t_eval):
    """
    Simulates the coupled Van der Pol oscillator system.

    Parameters:
    mu : array_like
        Nonlinearity parameters for each oscillator.
    adj_matrix : array_like
        Adjacency matrix describing the coupling between oscillators.
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
    result = solve_ivp(
        coupled_van_der_pol,
        t_span,
        y0,
        args=(mu, adj_matrix),
        t_eval=t_eval,
        method="RK45",
    )
    return result


if __name__ == "__main__":
    N = 3

    sampling = 5001

    seed = 0

    # Build the nearest-neighbor coupling graph for the coupled system.
    adj_matrix = np.diag(np.ones(N - 1), k=1) + np.diag(np.ones(N - 1), k=-1)

    rng = np.random.default_rng(seed=seed)

    # Sample one set of Van der Pol parameters and an initial condition for the training trajectory.
    mu = np.round(rng.uniform(1.0, 1.5, N), 2)

    y0_x = np.round(rng.uniform(-np.pi / 2, np.pi / 2, N), 2)
    y0_v = np.round(rng.uniform(-1.0, 1.0, N), 2)

    y0 = np.zeros(2 * N)
    for i in range(N):
        y0[2 * i] = y0_x[i]
        y0[2 * i + 1] = y0_v[i]

    dt = 0.01

    t_span = (0, (sampling - 1) * dt)
    t_eval = np.linspace(t_span[0], t_span[1], sampling)
    result = simulate_coupled_van_der_pol(mu, adj_matrix, y0, t_span, t_eval)
    
    test_data_num = 100
    
    for i in range(test_data_num):
        # Generate test trajectories by perturbing the nominal initial condition.
        rng_test = np.random.default_rng(i + 314871)  # Different random seed for test data

        y0_test_noise = rng_test.normal(-0.2, 0.2, N * 2)  # Add noise to initial conditions
        y0_test = y0 + y0_test_noise  # Perturb the original initial conditions with noise for test data
        result_test = simulate_coupled_van_der_pol(mu, adj_matrix, y0_test, t_span, t_eval)  # simulate test data
    print("Simulation completed.")
