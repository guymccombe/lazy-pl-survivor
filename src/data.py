from src.fixtures import get_upcoming_fixtures
from src.probabilities import win_probability, read_ratings_from_csv

import pandas as pd
import numpy as np


def prepare_data() -> pd.DataFrame:
    """
    Prepares the data for the survivor problem.
    Fixtures, skill ratings and win probabilities are calculated.
    Returns a DataFrame with index as teams and columns as gameweeks.
    """
    used_teams = get_used_teams()
    fixtures = get_upcoming_fixtures()
    ratings = read_ratings_from_csv()

    fixtures["home_att"] = fixtures["team_h"].map(ratings["attack"])
    fixtures["home_def"] = fixtures["team_h"].map(ratings["defense"])
    fixtures["away_att"] = fixtures["team_a"].map(ratings["attack"])
    fixtures["away_def"] = fixtures["team_a"].map(ratings["defense"])

    fixtures[["home_win_prob", "away_win_prob"]] = fixtures.apply(
        lambda x: win_probability(
            x["home_att"], x["home_def"], x["away_att"], x["away_def"]
        ),
        axis=1,
        result_type="expand",
    )

    home = (
        fixtures.groupby(["event", "team_h"], group_keys=True)[["home_win_prob"]]
        .max()
        .unstack()
    )
    away = (
        fixtures.groupby(["event", "team_a"], group_keys=True)[["away_win_prob"]]
        .max()
        .unstack()
    )
    maxes = np.maximum.reduce([df.fillna(0).to_numpy() for df in [home, away]])
    probabilities = pd.DataFrame(
        maxes, index=home.index, columns=home.columns.get_level_values(1)
    )
    if len(used_teams) > 0:
        probabilities = probabilities.drop(columns=used_teams)

    return probabilities.T


def get_used_teams(path: str = "../data/used.csv") -> set[str]:
    with open(path) as f:
        contents = f.read()
        if len(contents) == 0:
            return set()

        return {team.strip() for team in f.read().split(",")}
