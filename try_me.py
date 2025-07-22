# try_me.py

import matplotlib.pyplot as plt
from chaos.core import DynamicalSystem, Solver, SolverConfig
from chaos.systems import get_chaotic_system, list_chaotic_systems, ChaoticSystem

def main():
    print("Available chaotic systems:")
    for sys_name in list_chaotic_systems():
        print(f"- {sys_name}")

    # Select a system to simulate
    system_enum = ChaoticSystem.LORENZ
    func, params, initials = get_chaotic_system(system_enum)

    # Create DynamicalSystem instance
    system = DynamicalSystem(func=func, parameters=params)

    # Configure solver
    config = SolverConfig(t_span=(0, 40), time_step=0.01, cut_transient=0.1)

    # Create solver instance
    solver = Solver()

    # Solve with initial conditions from system definition
    result = solver.solve(system, initial_conditions=initials, config=config)

    # Plot results
    fig, ax = plt.subplots()

    variable_names = result.variable_names

    for name in variable_names:
        if hasattr(result, name):
            ax.plot(result.t, getattr(result, name), label=name)
        else:
            print(f"Warning: {name} not found in result.")

    ax.set_title(f"{system_enum.value.capitalize()} system simulation")
    ax.set_xlabel("Time")
    ax.set_ylabel("State variables")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
