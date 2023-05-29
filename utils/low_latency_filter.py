import numpy as np
from typing import Callable
from scipy.signal import butter, lfilter

class ButterworthFilter:
    def __init__(self, order, cutoff_freq, fs):
        self.order = order
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.b, self.a = butter(order, cutoff_freq, fs=fs, btype='lowpass')

    def filter(self, measurement):
        filtered_measurement = lfilter(self.b, self.a, measurement)
        return filtered_measurement[-1]

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
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def filter(self, measurement):
        self.buffer.append(measurement)

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        return sum(self.buffer) / len(self.buffer)


class CausalMedianFilter:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def filter(self, measurement):
        self.buffer.append(measurement)

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        return np.median(self.buffer)

# class UltraLowLatencyFilter:
#     def __init__(self, alpha, beta, order, cutoff_freq, fs):
#         self.alpha_beta_filter = AlphaBetaFilter(alpha, beta)
#         self.butterworth_filter = ButterworthFilter(order, cutoff_freq, fs)

#     def filter(self, measurement, dt):
#         alpha_beta_output = self.alpha_beta_filter.filter(measurement, dt)
#         return self.butterworth_filter.filter(alpha_beta_output)
    
# class UltraLowLatencyFilter:
#     def __init__(self, alpha, beta, buffer_size):
#         self.alpha_beta_filter = AlphaBetaFilter(alpha, beta)
#         self.median_filter = CausalMedianFilter(buffer_size)

#     def filter(self, measurement, dt):
#         alpha_beta_output = self.alpha_beta_filter.filter(measurement, dt)
#         return self.median_filter.filter(alpha_beta_output)
    
class UltraLowLatencyFilter:
    def __init__(self, alpha, beta, buffer_size):
        self.alpha_beta_filter = AlphaBetaFilter(alpha, beta)
        self.moving_average_filter = MovingAverageFilter(buffer_size)

    def filter(self, measurement, dt):
        alpha_beta_output = self.alpha_beta_filter.filter(measurement, dt)
        return self.moving_average_filter.filter(alpha_beta_output)
    
# class MovingAverageFilter:
#     def __init__(
#         self, buffer_size: int = 7, ave_func: Callable[[np.ndarray], float] = np.median
#     ):
#         self.buffer = np.zeros(buffer_size)
#         self.ave_func = ave_func

#     def filter(self, measurement):
#         self.buffer[0] = measurement
#         filtered_measurement = self.ave_func(self.buffer)
#         self.buffer[1:] = self.buffer[0:-1]

#         return filtered_measurement


# class UltraLowLatencyFilter:
#     def __init__(self, alpha, beta, buffer_size):
#         self.alpha_beta_filter = AlphaBetaFilter(alpha, beta)
#         self.moving_average_filter = MovingAverageFilter(buffer_size)

#     def filter(self, measurement, dt):
#         moving_ave_output = self.moving_average_filter.filter(measurement)
#         alpha_beta_output = self.alpha_beta_filter.filter(moving_ave_output, dt)
#         return alpha_beta_output
