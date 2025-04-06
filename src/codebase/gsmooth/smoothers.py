import jax.numpy as jnp
from functools import partial
import jax


class GradFilter(object):
    def __init__(self, step_size):
        """ Object that takes new gradients as an input, and
        outputs a de-noised version (estimate) """
        pass

    def smooth_gradient(self, noisy_gradient, output_variance=False):
        """ Takes in a gradient, and outputs a smoothed version of it """
        raise NotImplementedError

    def update(self, param, noisy_gradient, step_size):
        """ Takes a gradient step
        Input:
            param          : current parameter setting (JAX array)
            noisy_gradient : noisy gradient estimate
            step_size      : how big should the gradient step be?
        Output:
            new_param         : gradient updated parameter
            filtered_gradient : filtered gradient used for update
        """
        filtered_gradient = self.smooth_gradient(noisy_gradient)
        return param - step_size * filtered_gradient, filtered_gradient


class IdentityFilter(GradFilter):
    def __init__(self):
        """ Simple do-nothing filter --- for testing as a baseline """
        pass

    def smooth_gradient(self, noisy_gradient, output_variance=False):
        if output_variance:
            # Return zero variance since identity filter doesn't modify the gradient
            return noisy_gradient, jnp.zeros_like(noisy_gradient)
        return noisy_gradient


class AdamFilter(GradFilter):
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam uses a type of 'de-biased' exponential smoothing, so it can
        be viewed as a gradient smoother for stochastic gradient methods"""
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0.0  # First moment estimate
        self.v = 0.0  # Second moment estimate
        self.t = 0  # Timestep counter
        self.ave_step_sizes = []

    def smooth_gradient(self, noisy_gradient, output_variance=False):
        b1, b2 = self.beta1, self.beta2
        self.t = self.t + 1

        # Update biased first and second moment estimates
        self.m = b1 * self.m + (1.0 - b1) * noisy_gradient
        self.v = b2 * self.v + (1.0 - b2) * noisy_gradient ** 2

        # Compute bias-corrected estimates
        mhat = self.m / (1.0 - b1 ** self.t)
        vhat = self.v / (1.0 - b2 ** self.t)

        # Return smoothed gradient and variance if requested
        if output_variance:
            return mhat, vhat - mhat ** 2
        return mhat

    def update(self, param, noisy_gradient, step_size):
        mhat, vhat = self.smooth_gradient(noisy_gradient, output_variance=True)
        vhat += mhat ** 2  # Add squared mean back to get corrected vhat

        # Compute adaptive step size
        dparam = mhat / (jnp.sqrt(vhat) + self.eps)

        # In JAX, we can't modify lists in-place within a JIT-compiled function
        # So we record the average step size in a way that won't break JIT
        avg_step_size = jnp.mean(1.0 / (jnp.sqrt(vhat) + self.eps))
        self.ave_step_sizes.append(float(avg_step_size))  # Convert to Python scalar

        return param - step_size * dparam, (mhat, vhat)


class SGDMomentumFilter(GradFilter):
    def __init__(self, beta=0.9, eps=1e-8):
        """Momentum can also be considered a simple estimator of the true
        gradient. Beta is the "mass" parameter here"""
        self.beta = beta
        self.eps = eps
        self.v = 0.0  # Velocity
        self.t = 0  # Timestep counter

    def smooth_gradient(self, noisy_gradient, output_variance=False):
        self.v = self.beta * self.v - (1.0 - self.beta) * noisy_gradient
        self.t += 1

        if output_variance:
            # Momentum doesn't provide a variance estimate, return zeros
            return -self.v, jnp.zeros_like(noisy_gradient)
        return -self.v

    def update(self, param, noisy_gradient, step_size):
        smoothed_grad = self.smooth_gradient(noisy_gradient)
        return param - step_size * smoothed_grad, smoothed_grad


# JAX-optimized versions with explicit state handling for use with JIT

def create_adam_state(shape, beta1=0.9, beta2=0.999):
    """Create initial state for Adam optimizer"""
    return {
        'm': jnp.zeros(shape),  # First moment
        'v': jnp.zeros(shape),  # Second moment
        't': 0  # Timestep
    }


@jax.jit
def adam_update(state, grad, step_size, beta1=0.9, beta2=0.999, eps=1e-8):
    """JAX-optimized Adam update step that can be JIT-compiled"""
    # Unpack state
    m, v, t = state['m'], state['v'], state['t']

    # Update timestep
    t = t + 1

    # Update biased first and second moment estimates
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * grad ** 2

    # Compute bias-corrected estimates
    mhat = m / (1.0 - beta1 ** t)
    vhat = v / (1.0 - beta2 ** t)

    # Compute update
    update = mhat / (jnp.sqrt(vhat) + eps)

    # Return new state and update
    new_state = {'m': m, 'v': v, 't': t}
    return new_state, update


@jax.jit
def sgd_momentum_update(state, grad, step_size, beta=0.9):
    """JAX-optimized SGD with momentum update step that can be JIT-compiled"""
    # Unpack state
    v, t = state['v'], state['t']

    # Update velocity and timestep
    v = beta * v - (1.0 - beta) * grad
    t = t + 1

    # Return new state and update
    new_state = {'v': v, 't': t}
    return new_state, -v