# chaos/systems.py

from enum import Enum
from typing import Callable, Tuple, Dict, List
from .core import NamedValues
import numpy as np

# Define the enumeration of supported systems
class ChaoticSystem(Enum):
    LORENZ = "lorenz"
    ROSSLER = "rossler"
    CHEN = "chen"
    HALVORSEN = "halvorsen"
    THOMAS = "thomas"

# Registry to store system definitions
_SYSTEM_REGISTRY: Dict[ChaoticSystem, Callable[[], Tuple[
    Callable[[float, list, NamedValues], list],
    NamedValues,
    NamedValues
]]] = {}

# Decorator to register chaotic systems
def register_chaotic_system(name: ChaoticSystem):
    def decorator(fn: Callable[[], Tuple[Callable, NamedValues, NamedValues]]):
        _SYSTEM_REGISTRY[name] = fn
        return fn
    return decorator

@register_chaotic_system(ChaoticSystem.LORENZ)
def lorenz_system():
    def lorenz(t, state, p):
        x, y, z = state
        dx = p["sigma"] * (y - x)
        dy = x * (p["rho"] - z) - y
        dz = x * y - p["beta"] * z
        return [dx, dy, dz]

    params = NamedValues(values_dict={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3})
    initials = NamedValues(names=["x", "y", "z"], values=[1.0, 1.0, 1.0])
    return lorenz, params, initials

@register_chaotic_system(ChaoticSystem.ROSSLER)
def rossler_system():
    def rossler(t, state, p):
        x, y, z = state
        dx = -y - z
        dy = x + p["a"] * y
        dz = p["b"] + z * (x - p["c"])
        return [dx, dy, dz]

    params = NamedValues(values_dict={"a": 0.2, "b": 0.2, "c": 5.7})
    initials = NamedValues(names=["x", "y", "z"], values=[1.0, 1.0, 1.0])
    return rossler, params, initials

@register_chaotic_system(ChaoticSystem.CHEN)
def chen_system():
    def chen(t, state, p):
        x, y, z = state
        dx = p["a"] * (y - x)
        dy = (p["c"] - p["a"]) * x - x * z + p["c"] * y
        dz = x * y - p["b"] * z
        return [dx, dy, dz]

    params = NamedValues(values_dict={"a": 35.0, "b": 3.0, "c": 28.0})
    initials = NamedValues(names=["x", "y", "z"], values=[1.0, 1.0, 1.0])
    return chen, params, initials

@register_chaotic_system(ChaoticSystem.HALVORSEN)
def halvorsen_system():
    def halvorsen(t, state, p):
        x, y, z = state
        dx = -p["a"] * x - 4 * y - 4 * z - y ** 2
        dy = -p["a"] * y - 4 * z - 4 * x - z ** 2
        dz = -p["a"] * z - 4 * x - 4 * y - x ** 2
        return [dx, dy, dz]

    params = NamedValues(values_dict={"a": 1.4})
    initials = NamedValues(names=["x", "y", "z"], values=[1.0, 0.0, 0.0])
    return halvorsen, params, initials

@register_chaotic_system(ChaoticSystem.THOMAS)
def thomas_system():
    def thomas(t, state, p):
        x, y, z = state
        dx = -p["b"] * x + np.sin(y)
        dy = -p["b"] * y + np.sin(z)
        dz = -p["b"] * z + np.sin(x)
        return [dx, dy, dz]

    params = NamedValues(values_dict={"b": 0.208186})
    initials = NamedValues(names=["x", "y", "z"], values=[0.1, 0.0, -0.1])
    return thomas, params, initials

def get_chaotic_system(system: ChaoticSystem) -> Tuple[
    Callable[[float, list, NamedValues], list],
    NamedValues,
    NamedValues
]:
    """Retrieve function, parameters, and initials for a system."""
    try:
        return _SYSTEM_REGISTRY[system]()
    except KeyError:
        raise ValueError(f"Chaotic system '{system.value}' is not defined.")

def list_chaotic_systems() -> List[str]:
    """Return a list of available chaotic system names."""
    return [s.value for s in ChaoticSystem]
