import pandas as pd
from typing import Tuple
from scipy.stats import poisson


def read_ratings_from_csv(path: str = "../data/ratings.csv") -> pd.DataFrame:
    """
    Read Elevenify.com attack/defense ratings from a CSV file.
    """
    return pd.read_csv(path, index_col=0)


def __predict_xg(
    home_att: float, home_def: float, away_att: float, away_def: float
) -> Tuple[float, float]:
    """
    Predicts the expected goals created by each team.
    """

    # xG constants from fbref.com, 24/25 season
    HOME_XG = 1.712665406
    AWAY_XG = 1.351606805
    average_xg = (HOME_XG + AWAY_XG) / 2

    home_xg = HOME_XG / average_xg * home_att * away_def
    away_xg = AWAY_XG / average_xg * away_att * home_def

    return home_xg, away_xg


def __calculate_goal_distribution(xg: float, max_goals: int = 10) -> [float]:
    """
    Uses the Poisson distribution to calculate the probability of each number of goals scored by each team up to a maximum.
    """
    probabilities = [poisson.pmf(k=k, mu=xg) for k in range(max_goals + 1)]
    return probabilities


def __calculate_win_probability(
    winner_distribution: list[float], loser_distribution: list[float]
) -> float:
    """
    Calculates the probability of a team winning a match given the goal distributions of the winner and loser.
    Calculated by summing the probabilities of each winning score.
    """
    win_proba = 0
    for i in range(len(winner_distribution)):
        for j in range(i):
            # i always larger than j
            score_proba = winner_distribution[i] * loser_distribution[j]
            win_proba += score_proba
    return win_proba


def win_probability(
    home_att: float, home_def: float, away_att: float, away_def: float
) -> Tuple[float, float]:
    """
    Calculates the probability of each team winning the match.
    Returns a tuple of (home_win_prob, away_win_prob).
    """
    home_xg, away_xg = __predict_xg(home_att, home_def, away_att, away_def)
    home_goal_distribution = __calculate_goal_distribution(home_xg)
    away_goal_distribution = __calculate_goal_distribution(away_xg)

    home_win_prob = __calculate_win_probability(
        home_goal_distribution, away_goal_distribution
    )
    away_win_prob = __calculate_win_probability(
        away_goal_distribution, home_goal_distribution
    )

    return home_win_prob, away_win_prob
