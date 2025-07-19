# chaos/core.py

from typing import Callable, Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.integrate import solve_ivp
import json
import matplotlib.pyplot as plt


class NamedValues:
    """Immutable key-value container that preserves order and allows optional values."""

    def __init__(
        self,
        names: Optional[List[str]] = None,
        values: Optional[List[float]] = None,
        values_dict: Optional[Dict[str, Optional[float]]] = None,
    ):
        if values_dict is not None:
            self._keys = tuple(values_dict.keys())
            self._values = tuple(values_dict.get(k, None) for k in self._keys)
        elif names is not None:
            if values is None:
                values = [None] * len(names)
            if len(names) != len(values):
                raise ValueError("names and values must be the same length")
            self._keys = tuple(names)
            self._values = tuple(values)
        else:
            raise ValueError("Either values_dict or names must be provided")

        self._key_to_index = {k: i for i, k in enumerate(self._keys)}

    def __getitem__(self, key: str) -> Optional[float]:
        return self._values[self._key_to_index[key]]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def as_dict(self) -> Dict[str, Optional[float]]:
        return dict(zip(self._keys, self._values))

    def as_list(self) -> List[Optional[float]]:
        return list(self._values)

    def keys(self):
        return self._keys

    def values(self):
        return self._values

    def items(self):
        return zip(self._keys, self._values)

    def to_json(self) -> str:
        return json.dumps(self.as_dict())

    def validate(self):
        for k, v in self.items():
            if v is None:
                raise ValueError(f"Missing value for '{k}'")

    def __repr__(self):
        return f"NamedValues({self.as_dict()})"

    def __hash__(self):
        return hash((self._keys, self._values))

    def __eq__(self, other):
        return isinstance(other, NamedValues) and self._keys == other._keys and self._values == other._values


class SolverMethod(Enum):
    RK45 = "RK45"
    RK23 = "RK23"
    DOP853 = "DOP853"
    BDF = "BDF"
    LSODA = "LSODA"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class SolverConfig:
    t_span: Tuple[float, float] = (0.0, 50.0)
    time_step: float = 0.01
    method: SolverMethod = SolverMethod.RK45
    rtol: float = 1e-6
    atol: float = 1e-9
    cut_transient: float = 0.0  # percentage (0.0 to 1.0)

    def time_eval(self) -> np.ndarray:
        return np.arange(
            self.t_span[0],
            self.t_span[1] + self.time_step,
            self.time_step
        )

@dataclass
class DynamicalSystem:
    func: Callable[[float, List[float], NamedValues], List[float]]
    parameters: NamedValues

    def validate(self):
        self.parameters.validate()


@dataclass
class SimulationResult:
    t: np.ndarray
    y: np.ndarray
    variable_names: Tuple[str, ...]

    def as_array(self) -> np.ndarray:
        return self.y

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {name: self.y[i] for i, name in enumerate(self.variable_names)}

    def plot(self):
        import matplotlib.pyplot as plt

        for i, name in enumerate(self.variable_names):
            plt.plot(self.t, self.y[i], label=name)
        plt.xlabel("Time")
        plt.legend()
        plt.title("Dynamical System Simulation")
        plt.grid(True)
        plt.show()


class Solver:
    def solve(
        self,
        system: DynamicalSystem,
        initial_conditions: NamedValues,
        config: SolverConfig,
    ) -> SimulationResult:
        system.validate()
        initial_conditions.validate()

        t_eval = config.time_eval()

        def wrapped_func(t, y):
            return system.func(t, y, system.parameters)

        sol = solve_ivp(
            fun=wrapped_func,
            t_span=config.t_span,
            y0=initial_conditions.as_list(),
            t_eval=t_eval,
            method=config.method.value,
            rtol=config.rtol,
            atol=config.atol,
        )

        if not sol.success:
            raise RuntimeError(f"ODE Solver failed: {sol.message}")

        y_data = sol.y
        t_data = sol.t

        if config.cut_transient > 0:
            cut_index = int(len(t_data) * config.cut_transient)
            t_data = t_data[cut_index:]
            y_data = y_data[:, cut_index:]

        return SimulationResult(
            t=t_data,
            y=y_data,
            variable_names=tuple(initial_conditions.keys())
        )
