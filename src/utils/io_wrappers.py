from .decorators import torch_to_numpy, numpy_to_torch

def wrap_forward_numpy(model, device):
    """Wraps model.forward so it accepts NumPy inputs and returns NumPy outputs."""
    model.forward = torch_to_numpy(numpy_to_torch(device)(model.forward))


def wrap_decode_numpy(model, device):
    """Wraps model.decoder so it accepts NumPy inputs and returns NumPy outputs."""
    model.decode = torch_to_numpy(numpy_to_torch(device)(model.decode))

def wrap_encode_numpy(model, device):
    """Wraps model.decoder so it accepts NumPy inputs and returns NumPy outputs."""
    model.encode = torch_to_numpy(numpy_to_torch(device)(model.encode))
