import numpy as np
from typing import Callable


class AlphaBetaFilter:
    def __init__(self, alpha, beta):
        self.state = 0.0
        self.derivative = 0.0
        self.alpha = alpha
        self.beta = beta

    def filter(self, measurement, dt):
        predicted_state = self.state + dt * self.derivative
        residual = measurement - predicted_state

        self.state = predicted_state + self.alpha * residual
        self.derivative = self.derivative + self.beta * residual / dt

        return self.state


class MovingAverageFilter:
    def __init__(
        self, buffer_size: int = 7, ave_func: Callable[[np.ndarray], float] = np.median
    ):
        self.buffer = np.zeros(buffer_size)
        self.ave_func = ave_func

    def filter(self, measurement):
        self.buffer[0] = measurement
        filtered_measurement = self.ave_func(self.buffer)
        self.buffer[1:] = self.buffer[0:-1]

        return filtered_measurement


class UltraLowLatencyFilter:
    def __init__(self, alpha, beta, buffer_size):
        self.alpha_beta_filter = AlphaBetaFilter(alpha, beta)
        self.moving_average_filter = MovingAverageFilter(buffer_size)

    def filter(self, measurement, dt):
        moving_ave_output = self.moving_average_filter.filter(measurement)
        alpha_beta_output = self.alpha_beta_filter.filter(moving_ave_output, dt)
        return alpha_beta_output
