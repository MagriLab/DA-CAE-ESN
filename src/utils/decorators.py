import torch
from functools import wraps
import numpy as np
import functools
import time


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"{func.__name__}() in {run_time:.4f} secs")
        return value
    return wrapper_timer


def numpy_to_torch(device):
    """Convert NumPy inputs to torch.Tensor on the given device."""
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            def convert(x):
                if isinstance(x, np.ndarray):
                    return torch.from_numpy(x).float().to(device)
                return x
            args = tuple(convert(a) for a in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)
        return wrapped
    return decorator


def torch_to_numpy(func):
    """Convert torch.Tensor outputs to NumPy arrays."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if torch.is_tensor(result):
            return result.detach().cpu().numpy()
        elif isinstance(result, (tuple, list)):
            return type(result)(
                r.detach().cpu().numpy() if torch.is_tensor(r) else r
                for r in result
            )
        return result
    return wrapped
