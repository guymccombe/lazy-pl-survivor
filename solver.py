from math import exp

import numpy as np
import pandas as pd
from typing import Tuple
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
        decay: int = 0.84,
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
                for week in range(self.next_week, self.horizon):
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

    def solve(self): ...


def main():
    next_week, teams = boostrap_from_fpl_api()
    used_teams = get_used_teams()
    odds = get_match_odds()
    strengths = get_team_strengths()
    opt = SurvivorLinearOptimizer(next_week, teams, used_teams, odds, strengths)


if __name__ == "__main__":
    main()
