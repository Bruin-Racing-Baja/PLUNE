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


class UltraLowLatencyFilter:
    def __init__(self, alpha, beta, buffer_size):
        self.alpha_beta_filter = AlphaBetaFilter(alpha, beta)
        self.moving_average_filter = MovingAverageFilter(buffer_size)

    def filter(self, measurement, dt):
        alpha_beta_output = self.alpha_beta_filter.filter(measurement, dt)
        return self.moving_average_filter.filter(alpha_beta_output)

