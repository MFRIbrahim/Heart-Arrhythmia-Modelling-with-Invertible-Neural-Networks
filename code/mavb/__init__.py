
def print_solution(solution):
    x = solution["x"]
    objective = solution["adjusted objective"]
    rmse = solution["rmse"]
    print(f"RMSE: {rmse}")
    print(f"Adjusted Objective: {objective}")
    print(f"Atrial cycle length: {x[0]}")
    print(f"Inital time offset: {x[1]}")
    print(f"Conduction constant: {x[2]}")
