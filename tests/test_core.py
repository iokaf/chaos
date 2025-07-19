import pytest
import numpy as np
from chaos.core import (
    NamedValues,
    SolverConfig,
    SolverMethod,
    DynamicalSystem,
    Solver,
    SimulationResult
)

# --- NamedValues Tests ---

def test_named_values_init_and_access():
    nv = NamedValues(names=["a", "b"], values=[1.0, 2.0])
    assert nv["a"] == 1.0
    assert nv["b"] == 2.0
    assert nv.as_dict() == {"a": 1.0, "b": 2.0}
    assert nv.as_list() == [1.0, 2.0]

def test_named_values_init_from_dict():
    nv = NamedValues(values_dict={"x": 10.0, "y": 20.0})
    assert nv["x"] == 10.0
    assert nv["y"] == 20.0

def test_named_values_immutable():
    nv = NamedValues(values_dict={"x": 1})
    with pytest.raises(TypeError):
        nv._values[0] = 100

def test_named_values_validation():
    nv = NamedValues(values_dict={"a": 1.0, "b": None})
    with pytest.raises(ValueError):
        nv.validate()

def test_named_values_to_json():
    nv = NamedValues(values_dict={"alpha": 42.0, "beta": None})
    json_str = nv.to_json()
    assert '"alpha": 42.0' in json_str

def test_named_values_hashable():
    nv1 = NamedValues(values_dict={"x": 1.0, "y": 2.0})
    nv2 = NamedValues(values_dict={"x": 1.0, "y": 2.0})
    assert hash(nv1) == hash(nv2)
    assert nv1 == nv2

# --- SolverConfig Tests ---

def test_solver_config_defaults():
    config = SolverConfig()
    assert config.method == SolverMethod.RK45
    times = config.time_eval()
    assert isinstance(times, np.ndarray)
    assert len(times) > 0


def test_solver_config_time_eval_arange():
    config = SolverConfig(t_span=(0, 1), time_step=0.1)
    t_eval = config.time_eval()
    
    # Should start at t_span[0]
    assert np.isclose(t_eval[0], 0.0)
    
    # Should include endpoint approximately
    assert t_eval[-1] >= 1.0
    
    # Step size is consistent
    np.testing.assert_allclose(np.diff(t_eval), config.time_step)
    
    # Number of points matches expected count
    expected_points = int((config.t_span[1] - config.t_span[0]) / config.time_step) + 1
    assert len(t_eval) == expected_points


# --- DynamicalSystem + Solver Tests ---

def test_solver_lorenz_equivalent():
    def lorenz(t, state, p):
        x, y, z = state
        dx = p["sigma"] * (y - x)
        dy = x * (p["rho"] - z) - y
        dz = x * y - p["beta"] * z
        return [dx, dy, dz]

    params = NamedValues(values_dict={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3})
    initials = NamedValues(names=["x", "y", "z"], values=[1.0, 1.0, 1.0])

    system = DynamicalSystem(func=lorenz, parameters=params)
    solver = Solver()
    config = SolverConfig(t_span=(0, 1), time_step=0.01)
    result = solver.solve(system, initial_conditions=initials, config=config)

    assert isinstance(result, SimulationResult)
    assert result.y.shape[0] == 3
    assert result.t[0] >= 0
    assert "x" in result.variable_names

def test_solver_transient_cut():
    def dummy(t, state, p):
        return [0.0 for _ in state]

    initials = NamedValues(names=["x", "y"], values=[1.0, 2.0])
    params = NamedValues(values_dict={})
    system = DynamicalSystem(func=dummy, parameters=params)

    solver = Solver()
    config = SolverConfig(t_span=(0, 10), time_step=1.0, cut_transient=0.5)
    result = solver.solve(system, initial_conditions=initials, config=config)

    t_eval = np.arange(config.t_span[0], config.t_span[1] + config.time_step, config.time_step)
    expected_length = len(t_eval) - int(len(t_eval) * config.cut_transient)
    assert len(result.t) == expected_length

    # The dummy ODE returns zero derivatives, so state does not change:
    assert all((result.y[:, 0] == result.y[:, -1]))

