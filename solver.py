from math import exp

import numpy as np
import pandas as pd
from typing import Tuple, Iterable
import pulp
import requests


def boostrap_from_fpl_api() -> Tuple[int, dict[int, str]]:
    res = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    json = res.json()
    next_week = get_week(json)
    teams = get_teams(json)
    return next_week, teams


def get_week(json) -> int:
    for event in json["events"]:
        if event["is_next"]:
            return event["id"]


def get_teams(json) -> dict[int, str]:
    return {team["id"]: team["short_name"] for team in json["teams"]}


def get_used_teams() -> set[str]:
    with open("data/used.csv") as f:
        return {team.strip() for team in f.read().split(",")}


def get_match_odds() -> pd.DataFrame:
    return pd.read_csv("data/odds.csv", index_col=0)


def get_team_strengths() -> pd.DataFrame:
    return pd.read_csv("data/strength.csv", index_col=0)


def strength_heuristic(team: float, oppo: float, is_home: bool) -> float:
    """
    A close enough heuristic from playing around with odds, diffs and plots in Excel
    """
    return 0.9 / (1 + exp(-0.07 * (team - oppo))) - (0.1 * int(not is_home))


class SurvivorLinearOptimizer:
    def __init__(
        self,
        next_week: int,
        teams: dict[int, str],
        used_teams: set[str],
        odds: pd.DataFrame,
        strengths: pd.DataFrame,
        horizon: int = 12,
        decay: int = 0.975,
    ):
        self.next_week = next_week
        self.teams = teams
        self.used_teams = used_teams
        self.odds = odds
        self.strengths = strengths
        self.horizon = horizon
        self.decay = decay

        self.fixtures = self.get_fixtures()
        self.probas = self.calculate_probas()

    def get_fixtures(self) -> pd.DataFrame:
        res = requests.get("https://fantasy.premierleague.com/api/fixtures/")
        json = res.json()
        fixtures = []

        for id, name in self.teams.items():
            if name not in self.used_teams:
                row = {"id": id, "name": name}
                for week in range(self.next_week, self.horizon+2):
                    for match in json:
                        if match["event"] < week:
                            continue

                        if match["team_h"] == id:
                            assert week == match["event"]
                            row[str(week)] = (
                                self.teams[match["team_a"]],
                                True,
                            )  # (opponent, is_home)
                            break

                        elif match["team_a"] == id:
                            assert week == match["event"]
                            row[str(week)] = (
                                self.teams[match["team_h"]],
                                False,
                            )  # (opponent, is_home)
                            break

                fixtures.append(row)

        df = pd.DataFrame(fixtures)
        df = df.set_index("name", drop=True)
        return df

    def calculate_probas(self) -> pd.DataFrame:
        df = pd.DataFrame(index=self.fixtures.index, columns=self.fixtures.columns[1:])
        for week in df.columns:
            if week in self.odds.columns:
                df[week] = 1 / self.odds[week]
            else:
                for team in df.index:
                    opponent, is_home = self.fixtures.loc[team, week]
                    team_strength = self.strengths.loc[team, "strength"]
                    opponent_strength = self.strengths.loc[opponent, "strength"]
                    win_proba = max(
                        strength_heuristic(team_strength, opponent_strength, is_home), 0
                    )
                    df.loc[team, week] = win_proba

        return df

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
            prob += pulp.lpSum(x[t, w] for w, t in solution) <= len(solution) - 1

        return sorted(
            solutions, key=lambda solution_tuple: solution_tuple[0], reverse=True
        )

    def pretty_print_solutions(
        self, solutions: Iterable[Tuple[float, list[Tuple[str, str]]]]
    ) -> None:
        for i, (obj_value, solution) in enumerate(solutions, 1):
            print(f"\nSolution {i} (Total Expected Value: {obj_value:.4f}):")
            df = []
            for week, team in solution:
                oppo, is_home = self.fixtures.loc[team, week]
                proba = self.probas.loc[team, week]
                decay_factor = self.decay ** (int(week) - int(self.probas.columns[0]))
                ev = proba * decay_factor
                cum_ev = ev if len(df) == 0 else ev + df[-1]["cum_EV"]
                row = {
                    "pick": team,
                    "oppo": f"{oppo} ({'h' if is_home else 'a'})",
                    "proba": f"{proba:.4f}",
                    "EV": ev,
                    "cum_EV": cum_ev,
                }
                df.append(row)

            df = pd.DataFrame(data=df, index=self.probas.columns)
            print(df)


def main():
    next_week, teams = boostrap_from_fpl_api()
    used_teams = get_used_teams()
    odds = get_match_odds()
    strengths = get_team_strengths()
    opt = SurvivorLinearOptimizer(next_week, teams, used_teams, odds, strengths)
    solutions = opt.solve()
    opt.pretty_print_solutions(solutions)


if __name__ == "__main__":
    main()
