import autograd.numpy as np
import autograd

from utils import Timer


def power_itr(d):
    for k in range(d+1):
        for i in range(k+1):
            yield (k-i, i)


def poly(x, t, coeffs):
    n = coeffs.size
    degree = int((np.sqrt(8*n + 1) - 3) / 2)
    assert (degree + 2) * (degree + 1) / 2 == n

    return np.sum([coeffs[i] * x**xd * t**yd for i, (xd, yd) in enumerate(power_itr(degree))])


def trial_function(x, t, coeffs):
    return np.sin(np.pi * x) + x*(1-x)*t*poly(x, t, coeffs)


def u_analytic(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


def cost_function_generator(x_grid, t_grid):
    du_dt = autograd.elementwise_grad(trial_function, 1)
    d2u_dx2 = autograd.elementwise_grad(
        autograd.elementwise_grad(trial_function, 0), 0
    )

    def cost(coeffs, idxs=None):
        if idxs is None:
            return np.mean((du_dt(x_grid, t_grid, coeffs) - d2u_dx2(x_grid, t_grid, coeffs))**2)
        else:
            return np.mean((du_dt(x_grid[idxs], t_grid[idxs], coeffs) - d2u_dx2(x_grid[idxs], t_grid[idxs], coeffs))**2)

    return cost


def hess_descent(initial_coeffs, cost_gradient, cost_hessian):
    coeffs = initial_coeffs.copy()

    iterations = 5
    for i in range(iterations):
        grad = cost_gradient(coeffs)
        grad_len = grad.dot(grad)
        print(f"\r  {i+1}/{iterations}, |grad| = {grad_len:.3e}", end="")
        if grad_len < 1e-14:
            print()
            break

        coeffs -= np.linalg.inv(cost_hessian(coeffs)) @ grad
    else:
        print("\nyabe")

    return coeffs


def grad_descent(initial_coeffs, cost_gradient):
    coeffs = initial_coeffs.copy()
    velocity = np.zeros_like(coeffs)

    step_size = 0.00001
    mass = 0.9

    iterations = 400
    for i in range(iterations):
        grad = cost_gradient(coeffs)
        grad_len = grad.dot(grad)
        print(f"\r  {i+1}/{iterations}, |grad| = {grad_len:.3e}", end="")
        if grad_len < 1e-14:
            print()
            return coeffs
        elif grad_len > 1e14:
            print("\nGradient exploding, abort")
            return coeffs

        velocity = mass * velocity + step_size * grad
        coeffs -= velocity
    else:
        print()

    return coeffs


def get_optimal_prameters(degree, prev_coeffs, cost_function_test, cost_grad, cost_hess):
    n_coeffs = int((degree + 2) * (degree + 1) / 2)
    initial_coeffs = np.ones(n_coeffs)
    if not prev_coeffs is None:
        initial_coeffs[:prev_coeffs.size] = prev_coeffs
    print(f"  initial cost: {cost_function_test(initial_coeffs)}")

    timer = Timer()
    if degree < 2:
        optial_coeffs = hess_descent(initial_coeffs, cost_grad, cost_hess)
    else:
        optial_coeffs = grad_descent(initial_coeffs, cost_grad)

    print(f"  final cost: {cost_function_test(optial_coeffs)}")
    print(f"  time: {timer.get_pretty()}")

    return optial_coeffs


def main():
    # create test and train data
    Nx = 20
    Nt = 20
    T = np.log(100) / (np.pi**2)

    x_ray_test = np.linspace(0, 1, Nx)
    x_ray_train = (np.linspace(0, 1, Nx) + 0.5/Nx)[:-1]

    t_ray_test = np.linspace(0, T, Nt)
    t_ray_train = (np.linspace(0, T, Nt) + 0.5*T/Nt)[:-1]

    x_grid_train, t_grid_train = np.meshgrid(x_ray_train, t_ray_train)
    x_grid_test, t_grid_test = np.meshgrid(x_ray_test, t_ray_test)

    # create cost function and its derivatives
    cost_function_train = cost_function_generator(x_grid_train, t_grid_train)
    cost_function_test = cost_function_generator(x_grid_test, t_grid_test)

    cost_grad = autograd.grad(cost_function_train)
    cost_hess = autograd.hessian(cost_function_train)

    # find optimal parameters
    optial_coeffs = None
    for degree in [0, 1, 2, 3, 4, 5]:
        print(f"{degree=}")

        optial_coeffs = get_optimal_prameters(
            degree, optial_coeffs, cost_function_test, cost_grad, cost_hess
        )
        u_poly_grid = trial_function(x_grid_test, t_grid_test, optial_coeffs)
        u_ana_grid = u_analytic(x_grid_test, t_grid_test)

        np.savez(
            f"data/poly_solver/{degree=}.npz",
            x_ray=x_grid_test[0, :], t_ray=t_grid_test[:, 0],
            u_poly_grid=u_poly_grid, u_ana_grid=u_ana_grid
        )


if __name__ == "__main__":
    main()
