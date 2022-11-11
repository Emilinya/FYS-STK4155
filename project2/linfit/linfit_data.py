import numpy as np
from tqdm import tqdm

# hack to include code from utils
import sys
sys.path.insert(1, 'utils')
from gradient_descent import ScheduleType, StepType, GradSolver
from linfit_utils import LinFit, SplitData, Resampler


def f(x):
    term1 = 0.5*np.exp(-(9*x-7)**2/4.0)
    term2 = -0.2*np.exp(-(9*x-4)**2)
    return term1 + term2


def cost_function_generator(X_grid, y_ray):
    def cost_funciton(beta, lda=0):
        y_pred = X_grid @ beta
        if lda == 0:
            return np.sum((y_ray - y_pred)**2) / len(y_ray)
        else:
            return np.sum((y_ray - y_pred)**2) / len(y_ray) + lda * np.dot(beta, beta)
    return cost_funciton


def cost_function_gradient_generator(X_grid, y_ray):
    def cost_funciton_gradient(beta, lda=0, idxs=None):
        # idxs is used for stochastic gradient descent
        if idxs is None:
            y_pred = X_grid @ beta
            if lda == 0:
                return -2 * X_grid.T @ (y_ray - y_pred) / len(y_ray)
            else:
                return 2 * lda * beta - 2 * X_grid.T @ (y_ray - y_pred) / len(y_ray)
        else:
            y_pred = X_grid[idxs, :] @ beta
            if lda == 0:
                return -2 * X_grid[idxs, :].T @ (y_ray[idxs] - y_pred) / len(idxs)
            else:
                return 2 * lda * beta - 2 * X_grid[idxs, :].T @ (y_ray[idxs] - y_pred) / len(idxs)
    return cost_funciton_gradient


def get_betas(split_data, step_size, lda, grad_kwargs_list):
    if len(grad_kwargs_list) == 0:
        return

    cost_funciton = cost_function_generator(*split_data.train_data)
    cost_funciton_gradient = cost_function_gradient_generator(
        *split_data.train_data
    )

    parameter_count = split_data.train_data.X.shape[1]
    beta_grid = np.zeros((parameter_count, len(grad_kwargs_list)))

    grad_solver = GradSolver(
        poly_degree+1, cost_funciton, cost_funciton_gradient, *
        grad_kwargs_list[0]["type"]
    )

    # beta0 = np.random.randn(parameter_count)
    beta0 = np.ones(parameter_count)
    for i, grad_kwargs in enumerate(grad_kwargs_list):
        grad_solver.set_type(*grad_kwargs["type"])
        beta = grad_solver.solve(
            lda=lda, beta0=beta0, step_size=step_size, max_steps=100,
            **grad_kwargs.get("solve_args", {})
        )
        if beta is not None:
            beta_grid[:, i] = beta

    return beta_grid


def get_averages(poly_degree, x_ray, f_ray, step_size, grad_kwargs_list, lda=0, resamples=8):
    split_data = SplitData.from_1d_polynomial(poly_degree, x_ray, f_ray)
    cost_funciton = cost_function_generator(*split_data.test_data)

    for grad_kwargs in grad_kwargs_list:
        if (
            grad_kwargs["type"][1] == StepType.PLAIN_STOCHASTIC
            or grad_kwargs["type"][1] == StepType.MOMENTUM_STOCHASTIC
        ):
            grad_kwargs["solve_args"]["minibatch_count"] = int(
                len(split_data.train_data.y) /
                grad_kwargs["solve_args"]["minibatch_size"]
            )

    # analytic solution
    linfit = LinFit(split_data)
    analytic_beta = linfit.get_beta()

    resampler = Resampler(split_data, resamples)

    beta_cube = np.zeros(
        (resamples, poly_degree+1, len(grad_kwargs_list)))
    min_error_ray = np.full(len(grad_kwargs_list), float("infinity"))
    optimal_beta_grid = np.zeros((poly_degree+1, len(grad_kwargs_list)))
    MSE_grid = np.zeros((resamples, len(grad_kwargs_list)))

    for i in resampler.bootstrap():
        beta_cube[i, :, :] = get_betas(
            resampler.resampled_data, step_size, lda, grad_kwargs_list
        )
        error_list = [cost_funciton(beta) for beta in beta_cube[i, :, :].T]
        for j, (error, min_error) in enumerate(zip(error_list, min_error_ray)):
            if error < min_error:
                min_error_ray[j] = error
                optimal_beta_grid[:, j] = beta_cube[i, :, j]
        MSE_grid[i, :] = np.array(error_list)

    beta_mean_grid = np.mean(beta_cube, axis=0)
    beta_std_grid = np.std(beta_cube, axis=0)
    MSE_mean_ray = np.mean(MSE_grid, axis=0)
    MSE_std_ray = np.std(MSE_grid, axis=0)

    return analytic_beta, optimal_beta_grid, min_error_ray, beta_mean_grid, beta_std_grid, MSE_mean_ray, MSE_std_ray


def get_plain_momentum_params(poly_degree, x_ray, f_ray, schedule_type, step_size):
    mass_ray = np.linspace(0, 1, 50)

    optimal_mass = 0
    min_error = float("infinity")

    for mass in tqdm(mass_ray):
        grad_kwargs = [
            {"type": (schedule_type, StepType.MOMENTUM),
                "solve_args": {"mass": mass}},
        ]
        *_, MSE_mean_ray, _ = get_averages(
            poly_degree, x_ray, f_ray, step_size, grad_kwargs)

        if MSE_mean_ray[0] < min_error:
            optimal_mass = mass
            min_error = MSE_mean_ray[0]

    return optimal_mass


def get_stochastic_momentum_params(poly_degree, x_ray, f_ray, schedule_type, step_size):
    mass_ray = np.linspace(0, 1, 50)
    minibatch_ray = [3, 5, 8, 13, 21, 34, 55, 89, 144]

    optimal_mass = 0
    optimal_minibatch = 0
    min_error = float("infinity")

    for mass in tqdm(mass_ray):
        for minibatch in minibatch_ray:
            grad_kwargs = [
                {"type": (schedule_type, StepType.MOMENTUM_STOCHASTIC),
                    "solve_args": {"mass": mass, "minibatch_size": minibatch}},
            ]

            *_, MSE_mean_ray, _ = get_averages(
                poly_degree, x_ray, f_ray, step_size, grad_kwargs)

            if MSE_mean_ray[0] < min_error:
                optimal_mass = mass
                optimal_minibatch = minibatch
                min_error = MSE_mean_ray[0]

    return optimal_mass, optimal_minibatch


def analyze_schedule(poly_degree, x_ray, f_ray, schedule_type, hyper_params=None):
    if hyper_params is None:
        mass_plain = get_plain_momentum_params(
            poly_degree, x_ray, f_ray, schedule_type, step_size=0.25)
        mass_stochastic, minibatch_size = get_stochastic_momentum_params(
            poly_degree, x_ray, f_ray, schedule_type, step_size=0.25)
    else:
        mass_plain, mass_stochastic, minibatch_size = hyper_params
    print("got hyperparameters:", mass_plain, mass_stochastic, minibatch_size)

    grad_kwargs = [
        {"type": (schedule_type, StepType.PLAIN), "solve_args": {}},
        {"type": (schedule_type, StepType.MOMENTUM),
            "solve_args": {"mass": mass_plain}},
        {"type": (schedule_type, StepType.PLAIN_STOCHASTIC),
            "solve_args": {"minibatch_size": minibatch_size}},
        {"type": (schedule_type, StepType.MOMENTUM_STOCHASTIC),
            "solve_args": {"mass": mass_stochastic, "minibatch_size": minibatch_size}},
    ]

    step_size_ray = np.logspace(-3, 0, 100)

    best_beta_grid = np.zeros((poly_degree+1, len(grad_kwargs)+1))
    mean_grid = np.zeros((len(step_size_ray), len(grad_kwargs)))
    min_error_ray = np.full(len(grad_kwargs), float("infinity"))
    optimal_step_size_ray = np.zeros(len(grad_kwargs))
    std_grid = np.zeros_like(mean_grid)

    for i, step_size in enumerate(tqdm(step_size_ray)):
        analytic_beta, optimal_beta_grid, optimal_error_ray, _, _, MSE_mean_ray, MSE_std_ray = get_averages(
            poly_degree, x_ray, f_ray, step_size, grad_kwargs)

        if i == 0:
            best_beta_grid[:, 0] = analytic_beta

        for j, (min_err, err) in enumerate(zip(min_error_ray, optimal_error_ray)):
            if err < min_err:
                min_error_ray[j] = err
                optimal_step_size_ray[j] = step_size
                best_beta_grid[:, j+1] = optimal_beta_grid[:, j]

        mean_grid[i, :] = MSE_mean_ray
        std_grid[i, :] = MSE_std_ray

    schedule_str = str(schedule_type).split(".")[1]
    with open(f"linfit/data_{schedule_str}.dat", "w") as datafile:
        datafile.write(
            f"¤{best_beta_grid.shape}, best betas: analytic | plain | momentum | stochastic | stochastic momentum\n")
        np.savetxt(datafile, best_beta_grid)
        datafile.write(
            f"¤{optimal_step_size_ray.shape}, best step size: plain | momentum | stochastic | stochastic momentum\n")
        np.savetxt(datafile, optimal_step_size_ray)
        datafile.write(f"¤{step_size_ray.shape}, step sizes\n")
        np.savetxt(datafile, step_size_ray)
        datafile.write(f"¤{mean_grid.shape}, error means\n")
        np.savetxt(datafile, mean_grid)
        datafile.write(f"¤{std_grid.shape}, error stds\n")
        np.savetxt(datafile, std_grid)

    return grad_kwargs[np.argmin(min_error_ray)]["type"][1], (mass_plain, mass_stochastic, minibatch_size)


def OLS_analyzer(poly_degree, x_ray, f_ray, hyperparams=None):
    if hyperparams is None:
        hyperparams = [None]*4
    optimal_step_types = [None]*4

    print(" - ScheduleType.PLAIN - ")
    optimal_step_types[0], hyperparams[0] = analyze_schedule(
        poly_degree, x_ray, f_ray, ScheduleType.PLAIN, hyperparams[0])
    print(" - ScheduleType.ADAGRAD - ")
    optimal_step_types[1], hyperparams[1] = analyze_schedule(
        poly_degree, x_ray, f_ray, ScheduleType.ADAGRAD, hyperparams[1])
    print(" - ScheduleType.RMS_PROP - ")
    optimal_step_types[2], hyperparams[2] = analyze_schedule(
        poly_degree, x_ray, f_ray, ScheduleType.RMS_PROP, hyperparams[2])
    print(" - ScheduleType.ADAM - ")
    optimal_step_types[3], hyperparams[3] = analyze_schedule(
        poly_degree, x_ray, f_ray, ScheduleType.ADAM,  hyperparams[3])

    return optimal_step_types, hyperparams


def Ridge_analyzer(poly_degree, x_ray, f_ray, optimal_step_types, hyperparams):

    grad_kwargs = [
        {"type": (ScheduleType.PLAIN, optimal_step_types[0]),
            "solve_args": {}},
        {"type": (ScheduleType.ADAGRAD, optimal_step_types[1]),
            "solve_args": {}},
        {"type": (ScheduleType.RMS_PROP, optimal_step_types[2]),
            "solve_args": {}},
        {"type": (ScheduleType.ADAM, optimal_step_types[3]),
            "solve_args": {}},
    ]
    for grad_kwarg, hyperparam in zip(grad_kwargs, hyperparams):
        mass_plain, mass_stochastic, minibatch_size = hyperparam

        if grad_kwarg["type"][1] == StepType.MOMENTUM:
            grad_kwarg["solve_args"] = {"mass": mass_plain}
        elif grad_kwarg["type"][1] == StepType.PLAIN_STOCHASTIC:
            grad_kwarg["solve_args"] = {"minibatch_size": minibatch_size}
        elif grad_kwarg["type"][1] == StepType.MOMENTUM_STOCHASTIC:
            grad_kwarg["solve_args"] = {"mass": mass_stochastic, "minibatch_size": minibatch_size}

    step_size_ray = np.logspace(-3, 0, 100)
    lda_ray = np.logspace(-8, 0, 50)

    best_beta_grid = np.zeros((poly_degree+1, len(grad_kwargs)))
    min_error_ray = np.full(len(grad_kwargs), float("infinity"))

    optimal_step_size_ray = np.zeros(len(grad_kwargs))
    optimal_lda_ray = np.zeros(len(grad_kwargs))


    with tqdm(total=len(step_size_ray)*len(lda_ray)) as pbar:
        for step_size in step_size_ray:
            for lda in lda_ray:
                _, optimal_beta_grid, optimal_error_ray, _, _, _, _ = get_averages(
                poly_degree, x_ray, f_ray, step_size, grad_kwargs, lda=lda)

                for j, (min_err, err) in enumerate(zip(min_error_ray, optimal_error_ray)):
                    if err < min_err:
                        min_error_ray[j] = err
                        optimal_lda_ray[j] = lda
                        optimal_step_size_ray[j] = step_size
                        best_beta_grid[:, j] = optimal_beta_grid[:, j]
                pbar.update(1)

    with open(f"linfit/data_Ridge.dat", "w") as datafile:
        datafile.write(
            f"¤{best_beta_grid.shape}, best betas: plain | adagrad | rms prop | adam\n")
        np.savetxt(datafile, best_beta_grid)

        datafile.write(
            f"¤{optimal_step_size_ray.shape}, best step size: plain | adagrad | rms prop | adam\n")
        np.savetxt(datafile, optimal_step_size_ray)

        datafile.write(
            f"¤{optimal_lda_ray.shape}, best lambda: plain | adagrad | rms prop | adam\n")
        np.savetxt(datafile, optimal_lda_ray)
    
        datafile.write(f"¤{step_size_ray.shape}, step sizes\n")
        np.savetxt(datafile, step_size_ray)

        datafile.write(f"¤{lda_ray.shape}, lambdas\n")
        np.savetxt(datafile, lda_ray)

if __name__ == "__main__":
    np.random.seed(8008)

    poly_degree = 10
    x_ray = np.linspace(0, 1, 500)
    f_ray = f(x_ray)

    # reuse calculated hyperparams from previous runs
    hyperparams = [
        (0.8979591836734693, 0.8775510204081632, 89),
        (0.9183673469387754, 0.44897959183673464, 21),
        (0.36734693877551017, 0.4897959183673469, 21),
        (1.0, 0.632653061224489, 144)
    ]

    # reuse optimal step type from previous runs
    optimal_step_types = [
        StepType.MOMENTUM,
        StepType.MOMENTUM_STOCHASTIC,
        StepType.MOMENTUM_STOCHASTIC,
        StepType.MOMENTUM_STOCHASTIC,
    ]

    # # start by finding optimal step type using OLS regression
    # optimal_step_types, hyperparams = OLS_analyzer(
    #     poly_degree, x_ray, f_ray, hyperparams)

    # see if Ridge regression can improve approximation for optimal step types
    # Ridge_analyzer(poly_degree, x_ray, f_ray, optimal_step_types, hyperparams)
