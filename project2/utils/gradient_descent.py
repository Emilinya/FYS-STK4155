from __future__ import annotations
import numpy as np
from enum import Enum, auto
from typing import Callable, Any


class StepType(Enum):
    PLAIN = auto()
    MOMENTUM = auto()
    PLAIN_STOCHASTIC = auto()
    MOMENTUM_STOCHASTIC = auto()


class ScheduleType(Enum):
    PLAIN = auto()
    ADAGRAD = auto()
    RMS_PROP = auto()
    ADAM = auto()


class GradSolver():
    def __init__(
        self, beta_count: int, cost_func: Callable[[np.ndarray], float],
        cost_func_grad: Callable[[np.ndarray], np.ndarray], schedule_type: ScheduleType,
        step_type: StepType
    ):
        self.n = beta_count
        self.C = cost_func
        self.grad_C = cost_func_grad
        self.schedule_type = schedule_type
        self.step_type = step_type

    def set_schedule_type(self, schedule_type: ScheduleType):
        self.schedule_type = schedule_type

    def set_step_type(self, step_type: StepType):
        self.step_type = step_type

    def set_type(self, schedule_type: ScheduleType, step_type: StepType):
        self.set_step_type(step_type)
        self.set_schedule_type(schedule_type)

    # there sure are many combinations of step types and schedulers
    def solve(
        self, *, lda=0, beta0=None, step_size=None, mass=None,
        minibatch_count=None, minibatch_size=None, max_steps=None
    ):
        if step_size is None:
            print("GradSolver.solve: missing step_size")
            return None
        if max_steps is None:
            print("GradSolver.solve: missing max_steps")
            return None

        if self.schedule_type == ScheduleType.PLAIN:
            if self.step_type == StepType.PLAIN:
                return self.plain_plain_solve(lda, beta0, step_size, max_steps)
            if self.step_type == StepType.MOMENTUM:
                return self.plain_momentum_solve(lda, beta0, step_size, mass, max_steps)
            if self.step_type == StepType.PLAIN_STOCHASTIC:
                return self.plain_plain_stochastic_solve(
                    lda, beta0, step_size, minibatch_count, minibatch_size, max_steps
                )
            if self.step_type == StepType.MOMENTUM_STOCHASTIC:
                return self.plain_momentum_stochastic_solve(
                    lda, beta0, step_size, mass, minibatch_count, minibatch_size, max_steps
                )

        if self.schedule_type == ScheduleType.ADAGRAD:
            if self.step_type == StepType.PLAIN:
                return self.adagrad_plain_solve(lda, beta0, step_size, max_steps)
            if self.step_type == StepType.MOMENTUM:
                return self.adagrad_momentum_solve(lda, beta0, step_size, mass, max_steps)
            if self.step_type == StepType.PLAIN_STOCHASTIC:
                return self.adagrad_plain_stochastic_solve(
                    lda, beta0, step_size, minibatch_count, minibatch_size, max_steps
                )
            if self.step_type == StepType.MOMENTUM_STOCHASTIC:
                return self.adagrad_momentum_stochastic_solve(
                    lda, beta0, step_size, mass, minibatch_count, minibatch_size, max_steps
                )

        if self.schedule_type == ScheduleType.RMS_PROP:
            if self.step_type == StepType.PLAIN:
                return self.rms_prop_plain_solve(lda, beta0, step_size, max_steps)
            if self.step_type == StepType.MOMENTUM:
                return self.rms_prop_momentum_solve(lda, beta0, step_size, mass, max_steps)
            if self.step_type == StepType.PLAIN_STOCHASTIC:
                return self.rms_prop_plain_stochastic_solve(
                    lda, beta0, step_size, minibatch_count, minibatch_size, max_steps
                )
            if self.step_type == StepType.MOMENTUM_STOCHASTIC:
                return self.rms_prop_momentum_stochastic_solve(
                    lda, beta0, step_size, mass, minibatch_count, minibatch_size, max_steps
                )

        if self.schedule_type == ScheduleType.ADAM:
            if self.step_type == StepType.PLAIN:
                return self.adam_plain_solve(lda, beta0, step_size, max_steps)
            if self.step_type == StepType.MOMENTUM:
                return self.adam_momentum_solve(lda, beta0, step_size, mass, max_steps)
            if self.step_type == StepType.PLAIN_STOCHASTIC:
                return self.adam_plain_stochastic_solve(
                    lda, beta0, step_size, minibatch_count, minibatch_size, max_steps
                )
            if self.step_type == StepType.MOMENTUM_STOCHASTIC:
                return self.adam_momentum_stochastic_solve(
                    lda, beta0, step_size, mass, minibatch_count, minibatch_size, max_steps
                )

        print(
            f"unimplemented ScheduleType/StepType: {self.schedule_type}/{self.step_type}"
        )
        return None

    # -- plain --

    def plain_plain_solve(self, lda: float, beta0: np.ndarray, step_size: float, max_steps: int):
        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()

        for _ in range(max_steps):
            grad = self.grad_C(beta, lda=lda)
            grad_size = np.dot(grad, grad)
            if grad_size < 1e-14:
                return beta
            if grad_size > 1e100:
                return None
            beta += -step_size * grad

        return beta

    def plain_momentum_solve(self, lda: float, beta0: np.ndarray, step_size: float, mass: float, max_steps: int):
        if mass is None:
            print("GradSolver.plain_momentum_solve: missing mass")
            return None

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()
        velocity = np.zeros_like(beta)

        for _ in range(max_steps):
            grad = self.grad_C(beta, lda=lda)
            grad_size = np.dot(grad, grad)
            if grad_size < 1e-14:
                return beta
            if grad_size > 1e100:
                return None

            velocity = mass * velocity + step_size * grad
            beta += -velocity

        return beta

    def plain_plain_stochastic_solve(
        self, lda: float, beta0: np.ndarray, step_size: float,
        minibatch_count: int, minibatch_size: int, max_steps: int
    ):
        if minibatch_count is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_count")
            return None
        if minibatch_size is None:
            print("GradSolver.plain_plain_stochastic_solve: missing minibatch_size")
            return None

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()

        epochs = int(np.ceil(max_steps / minibatch_count))
        for _ in range(epochs):
            for _ in range(minibatch_count):
                idx = minibatch_size*np.random.randint(minibatch_count)
                grad = self.grad_C(beta, lda=lda, idxs=np.array(
                    range(idx, idx+minibatch_size)))

                if np.dot(grad, grad) > 1e100:
                    return None

                beta += -step_size * grad

            # calculate full gradient to see if we are done
            grad = self.grad_C(beta, lda=lda)
            if np.dot(grad, grad) < 1e-14:
                return beta

        return beta

    def plain_momentum_stochastic_solve(
        self, lda: float, beta0: np.ndarray, step_size: float, mass: float,
        minibatch_count: int, minibatch_size: int, max_steps: int
    ):
        if mass is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing mass")
            return None
        if minibatch_count is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_count")
            return None
        if minibatch_size is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_size")
            return None

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()
        velocity = np.zeros_like(beta)

        epochs = int(np.ceil(max_steps / minibatch_count))
        for _ in range(epochs):
            for _ in range(minibatch_count):
                idx = minibatch_size*np.random.randint(minibatch_count)
                grad = self.grad_C(beta, lda=lda, idxs=np.array(
                    range(idx, idx+minibatch_size)))

                if np.dot(grad, grad) > 1e100:
                    return None

                velocity = mass * velocity + step_size * grad
                beta += -velocity

            # calculate full gradient to see if we are done
            grad = self.grad_C(beta, lda=lda)
            if np.dot(grad, grad) < 1e-14:
                return beta

        return beta

    # -- adagrad --

    def adagrad_plain_solve(self, lda: float, beta0: np.ndarray, step_size: float, max_steps: int):
        if max_steps is None:
            print("GradSolver.adagrad_plain_solve: missing max_steps")
            return None

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()

        G = np.zeros((self.n, self.n))
        for _ in range(max_steps):
            grad = self.grad_C(beta, lda=lda)
            grad_size = np.dot(grad, grad)
            if grad_size < 1e-14:
                return beta
            if grad_size > 1e100:
                return None

            G += np.outer(grad, grad)
            Ginv = step_size / (1e-10 + np.sqrt(np.diagonal(G)))

            beta += -np.multiply(Ginv, grad)
        return beta

    def adagrad_momentum_solve(self, lda: float, beta0: np.ndarray, step_size: float, mass: float, max_steps: int):
        if mass is None:
            print("GradSolver.plain_momentum_solve: missing mass")
            return None

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()
        velocity = np.zeros_like(beta)

        G = np.zeros((self.n, self.n))
        for _ in range(max_steps):
            grad = self.grad_C(beta, lda=lda)
            grad_size = np.dot(grad, grad)
            if grad_size < 1e-14:
                return beta
            if grad_size > 1e100:
                return None

            velocity = mass * velocity + step_size * grad

            G += np.outer(velocity, velocity)
            Ginv = step_size / (1e-10 + np.sqrt(np.diagonal(G)))

            beta += -np.multiply(Ginv, velocity)

        return beta

    def adagrad_plain_stochastic_solve(
        self, lda: float, beta0: np.ndarray, step_size: float,
        minibatch_count: int, minibatch_size: int, max_steps: int
    ):
        if minibatch_count is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_count")
            return None
        if minibatch_size is None:
            print("GradSolver.plain_plain_stochastic_solve: missing minibatch_size")
            return None

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()

        epochs = int(np.ceil(max_steps / minibatch_count))
        for _ in range(epochs):
            G = np.zeros((self.n, self.n))
            for _ in range(minibatch_count):
                idx = minibatch_size*np.random.randint(minibatch_count)
                grad = self.grad_C(beta, lda=lda, idxs=np.array(
                    range(idx, idx+minibatch_size)))

                if np.dot(grad, grad) > 1e100:
                    return None

                G += np.outer(grad, grad)
                Ginv = step_size / (1e-10 + np.sqrt(np.diagonal(G)))

                beta += -np.multiply(Ginv, grad)

            # calculate full gradient to see if we are done
            grad = self.grad_C(beta, lda=lda)
            if np.dot(grad, grad) < 1e-14:
                return beta

        return beta

    def adagrad_momentum_stochastic_solve(
        self, lda: float, beta0: np.ndarray, step_size: float, mass: float,
        minibatch_count: int, minibatch_size: int, max_steps: int
    ):
        if mass is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing mass")
            return None
        if minibatch_count is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_count")
            return None
        if minibatch_size is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_size")
            return None

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()
        velocity = np.zeros_like(beta)

        epochs = int(np.ceil(max_steps / minibatch_count))
        for _ in range(epochs):
            G = np.zeros((self.n, self.n))
            for _ in range(minibatch_count):
                idx = minibatch_size*np.random.randint(minibatch_count)
                grad = self.grad_C(beta, lda=lda, idxs=np.array(
                    range(idx, idx+minibatch_size)))

                if np.dot(grad, grad) > 1e100:
                    return None

                velocity = mass * velocity + step_size * grad

                G += np.outer(velocity, velocity)
                Ginv = step_size / (1e-10 + np.sqrt(np.diagonal(G)))

                beta += -np.multiply(Ginv, velocity)

            # calculate full gradient to see if we are done
            grad = self.grad_C(beta, lda=lda)
            if np.dot(grad, grad) < 1e-14:
                return beta

        return beta

    # -- rms prop --

    def rms_prop_plain_solve(self, lda: float, beta0: np.ndarray, step_size: float, max_steps: int):
        if max_steps is None:
            print("GradSolver.adagrad_plain_solve: missing max_steps")
            return None

        memlif1 = 0.9

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()

        s = np.zeros((self.n, self.n))
        for _ in range(max_steps):
            grad = self.grad_C(beta, lda=lda)
            grad_size = np.dot(grad, grad)
            if grad_size < 1e-14:
                return beta
            if grad_size > 1e100:
                return None

            s = memlif1*s + (1 - memlif1)*np.outer(grad, grad)
            sinv = step_size / (1e-10 + np.sqrt(np.diagonal(s)))

            beta += -np.multiply(sinv, grad)
        return beta

    def rms_prop_momentum_solve(self, lda: float, beta0: np.ndarray, step_size: float, mass: float, max_steps: int):
        if mass is None:
            print("GradSolver.plain_momentum_solve: missing mass")
            return None

        memlif1 = 0.9

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()
        velocity = np.zeros_like(beta)

        s = np.zeros((self.n, self.n))
        for _ in range(max_steps):
            grad = self.grad_C(beta, lda=lda)
            grad_size = np.dot(grad, grad)
            if grad_size < 1e-14:
                return beta
            if grad_size > 1e100:
                return None

            velocity = mass * velocity + step_size * grad

            s = memlif1*s + (1 - memlif1)*np.outer(velocity, velocity)
            sinv = step_size / (1e-10 + np.sqrt(np.diagonal(s)))

            beta += -np.multiply(sinv, velocity)

        return beta

    def rms_prop_plain_stochastic_solve(
        self, lda: float, beta0: np.ndarray, step_size: float,
        minibatch_count: int, minibatch_size: int, max_steps: int
    ):
        if minibatch_count is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_count")
            return None
        if minibatch_size is None:
            print("GradSolver.plain_plain_stochastic_solve: missing minibatch_size")
            return None

        memlif1 = 0.9

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()

        epochs = int(np.ceil(max_steps / minibatch_count))
        for _ in range(epochs):
            s = np.zeros((self.n, self.n))
            for _ in range(minibatch_count):
                idx = minibatch_size*np.random.randint(minibatch_count)
                grad = self.grad_C(beta, lda=lda, idxs=np.array(
                    range(idx, idx+minibatch_size)))

                if np.dot(grad, grad) > 1e100:
                    return None

                s = memlif1*s + (1 - memlif1)*np.outer(grad, grad)
                sinv = step_size / (1e-10 + np.sqrt(np.diagonal(s)))

                beta += -np.multiply(sinv, grad)

            # calculate full gradient to see if we are done
            grad = self.grad_C(beta, lda=lda)
            if np.dot(grad, grad) < 1e-14:
                return beta

        return beta

    def rms_prop_momentum_stochastic_solve(
        self, lda: float, beta0: np.ndarray, step_size: float, mass: float,
        minibatch_count: int, minibatch_size: int, max_steps: int
    ):
        if mass is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing mass")
            return None
        if minibatch_count is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_count")
            return None
        if minibatch_size is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_size")
            return None

        memlif1 = 0.9

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()
        velocity = np.zeros_like(beta)

        epochs = int(np.ceil(max_steps / minibatch_count))
        for _ in range(epochs):
            s = np.zeros((self.n, self.n))
            for _ in range(minibatch_count):
                idx = minibatch_size*np.random.randint(minibatch_count)
                grad = self.grad_C(beta, lda=lda, idxs=np.array(
                    range(idx, idx+minibatch_size)))

                if np.dot(grad, grad) > 1e100:
                    return None

                velocity = mass * velocity + step_size * grad

                s = memlif1*s + (1 - memlif1)*np.outer(velocity, velocity)
                sinv = step_size / (1e-10 + np.sqrt(np.diagonal(s)))

                beta += -np.multiply(sinv, velocity)

            # calculate full gradient to see if we are done
            grad = self.grad_C(beta, lda=lda)
            if np.dot(grad, grad) < 1e-14:
                return beta

        return beta

    # -- adam --

    def adam_plain_solve(self, lda: float, beta0: np.ndarray, step_size: float, max_steps: int):
        if max_steps is None:
            print("GradSolver.adagrad_plain_solve: missing max_steps")
            return None

        memlif1 = 0.9
        memlif2 = 0.99

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()

        m = np.zeros(self.n)
        s = np.zeros((self.n, self.n))
        for t in range(1, max_steps+1):
            grad = self.grad_C(beta, lda=lda)
            grad_size = np.dot(grad, grad)
            if grad_size < 1e-14:
                return beta
            if grad_size > 1e100:
                return None

            m = memlif1*m + (1 - memlif1)*grad
            s = memlif2*s + (1 - memlif2)*np.outer(grad, grad)
            m = m/(1 - memlif1**t)
            s = s/(1 - memlif2**t)
        
            sinv = step_size / (1e-10 + np.sqrt(np.diagonal(s)))

            beta += -np.multiply(sinv, m)
        return beta

    def adam_momentum_solve(self, lda: float, beta0: np.ndarray, step_size: float, mass: float, max_steps: int):
        if mass is None:
            print("GradSolver.plain_momentum_solve: missing mass")
            return None

        memlif1 = 0.9
        memlif2 = 0.99

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()
        velocity = np.zeros_like(beta)

        m = np.zeros(self.n)
        s = np.zeros((self.n, self.n))
        for t in range(1, max_steps+1):
            grad = self.grad_C(beta, lda=lda)
            grad_size = np.dot(grad, grad)
            if grad_size < 1e-14:
                return beta
            if grad_size > 1e100:
                return None

            velocity = mass * velocity + step_size * grad

            m = memlif1*m + (1 - memlif1)*velocity
            s = memlif2*s + (1 - memlif2)*np.outer(velocity, velocity)
            m = m/(1 - memlif1**t)
            s = s/(1 - memlif2**t)
        
            sinv = step_size / (1e-10 + np.sqrt(np.diagonal(s)))

            beta += -np.multiply(sinv, m)

        return beta

    def adam_plain_stochastic_solve(
        self, lda: float, beta0: np.ndarray, step_size: float,
        minibatch_count: int, minibatch_size: int, max_steps: int
    ):
        if minibatch_count is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_count")
            return None
        if minibatch_size is None:
            print("GradSolver.plain_plain_stochastic_solve: missing minibatch_size")
            return None

        memlif1 = 0.9
        memlif2 = 0.99

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()

        epochs = int(np.ceil(max_steps / minibatch_count))
        for _ in range(epochs):
            m = np.zeros(self.n)
            s = np.zeros((self.n, self.n))
            for t in range(1, minibatch_count+1):
                idx = minibatch_size*np.random.randint(minibatch_count)
                grad = self.grad_C(beta, lda=lda, idxs=np.array(
                    range(idx, idx+minibatch_size)))

                if np.dot(grad, grad) > 1e100:
                    return None

                m = memlif1*m + (1 - memlif1)*grad
                s = memlif2*s + (1 - memlif2)*np.outer(grad, grad)
                m = m/(1 - memlif1**t)
                s = s/(1 - memlif2**t)
            
                sinv = step_size / (1e-10 + np.sqrt(np.diagonal(s)))

                beta += -np.multiply(sinv, m)

            # calculate full gradient to see if we are done
            grad = self.grad_C(beta, lda=lda)
            if np.dot(grad, grad) < 1e-14:
                return beta

        return beta

    def adam_momentum_stochastic_solve(
        self, lda: float, beta0: np.ndarray, step_size: float, mass: float,
        minibatch_count: int, minibatch_size: int, max_steps: int
    ):
        if mass is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing mass")
            return None
        if minibatch_count is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_count")
            return None
        if minibatch_size is None:
            print("GradSolver.plain_momentum_stochastic_solve: missing minibatch_size")
            return None

        memlif1 = 0.9
        memlif2 = 0.99

        if beta0 is None:
            beta: np.ndarray[Any, np.dtype[np.float64]
                             ] = np.random.randn(self.n)
        else:
            beta = beta0.copy()
        velocity = np.zeros_like(beta)

        epochs = int(np.ceil(max_steps / minibatch_count))
        for _ in range(epochs):
            m = np.zeros(self.n)
            s = np.zeros((self.n, self.n))
            for t in range(1, minibatch_count+1):
                idx = minibatch_size*np.random.randint(minibatch_count)
                grad = self.grad_C(beta, lda=lda, idxs=np.array(
                    range(idx, idx+minibatch_size)))

                if np.dot(grad, grad) > 1e100:
                    return None

                velocity = mass * velocity + step_size * grad

                m = memlif1*m + (1 - memlif1)*velocity
                s = memlif2*s + (1 - memlif2)*np.outer(velocity, velocity)
                m = m/(1 - memlif1**t)
                s = s/(1 - memlif2**t)
            
                sinv = step_size / (1e-10 + np.sqrt(np.diagonal(s)))

                beta += -np.multiply(sinv, m)

            # calculate full gradient to see if we are done
            grad = self.grad_C(beta, lda=lda)
            if np.dot(grad, grad) < 1e-14:
                return beta

        return beta
