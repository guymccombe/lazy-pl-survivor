import pandas as pd
import pulp
from typing import Tuple, Iterable
from src.data import prepare_data


class SurvivorLinearOptimizer:
    def __init__(
        self,
        data: pd.DataFrame,
        decay: int = 0.88,
    ):
        self.probas = data
        self.decay = decay

    def solve(
        self, n_solutions: int = 5
    ) -> Iterable[Tuple[float, list[Tuple[str, str]]]]:
        prob = pulp.LpProblem("Survivor_Optimization", pulp.LpMaximize)
        weeks: list[str] = self.probas.columns
        teams: list[str] = self.probas.index

        # Decision variables
        x = pulp.LpVariable.dicts(
            "pick", ((t, w) for t in teams for w in weeks), cat="Binary"
        )

        # Objective function
        prob += pulp.lpSum(
            self.probas.loc[t, w] * x[t, w] * (self.decay ** (int(w) - int(weeks[0])))
            for t in teams
            for w in weeks
        )

        # Constraints
        # Pick one team per week
        for w in weeks:
            prob += pulp.lpSum(x[t, w] for t in teams) == 1

        # Pick each team at most once
        for t in teams:
            prob += pulp.lpSum(x[t, w] for w in weeks) <= 1

        solutions = []
        for i in range(n_solutions):
            # Solve the problem
            prob.solve()

            if pulp.LpStatus[prob.status] != "Optimal":
                break

            # Extract the solution
            solution = []
            for w in weeks:
                for t in teams:
                    if (
                        x[t, w].value() > 0.5
                    ):  # Using > 0.5 instead of == 1 to account for floating-point imprecision
                        solution.append((w, t))
                        break

            # Calculate the objective value
            obj_value = sum(
                self.probas.loc[t, w] * (self.decay ** (int(w) - int(weeks[0])))
                for w, t in solution
            )

            solutions.append((obj_value, solution))

            # Add a constraint to exclude this solution in the next iteration
            # must have at least horizon//2 differences
            required_differences = len(solution) // 2
            prob += (
                pulp.lpSum(x[t, w] for w, t in solution)
                <= len(solution) - required_differences
            )

        return sorted(
            solutions, key=lambda solution_tuple: solution_tuple[0], reverse=True
        )

    def pretty_print_solutions(
        self, solutions: Iterable[Tuple[float, list[Tuple[str, str]]]]
    ) -> None:
        for i, (obj_value, solution) in enumerate(solutions, 1):
            print(f"\nSolution {i} (Total Expected Value: {obj_value:.4f}):")
            df = []
            proba = 1
            probability_to_reach = 1
            for week, team in solution:
                probability_to_reach *= 1 if len(df) == 0 else proba
                proba = self.probas.loc[team, week]
                decay_factor = self.decay ** (int(week) - int(self.probas.columns[0]))
                ev = proba * decay_factor
                cum_ev = ev if len(df) == 0 else ev + df[-1]["cum_EV"]
                row = {
                    "pick": team,
                    "win_probability": f"{100*proba:.2f}%",
                    "EV": ev,
                    "cum_EV": cum_ev,
                    "probability_to_reach": f"{100*probability_to_reach:.2f}%",
                }
                df.append(row)

            df = pd.DataFrame(data=df, index=self.probas.columns)
            print(df)


def main():
    data = prepare_data()
    opt = SurvivorLinearOptimizer(data)
    solutions = opt.solve()
    opt.pretty_print_solutions(solutions)


if __name__ == "__main__":
    main()
