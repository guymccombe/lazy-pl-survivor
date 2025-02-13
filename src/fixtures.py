import requests
from typing import Tuple
import pandas as pd


def __boostrap_from_fpl_api() -> Tuple[int, dict[int, str]]:
    """
    Fetches the index of the next gameweek and a mapping from team IDs to team names from the FPL API.
    """
    res = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    json = res.json()
    next_week = float("inf")
    for event in json["events"]:
        if event["is_next"]:
            next_week = event["id"]
    teams = {team["id"]: team["short_name"] for team in json["teams"]}
    return next_week, teams


def __all_fixtures() -> pd.DataFrame:
    """
    Fetches the fixtures from the FPL API.
    """
    res = requests.get("https://fantasy.premierleague.com/api/fixtures/")
    json = res.json()
    return pd.DataFrame(json)


def get_upcoming_fixtures() -> pd.DataFrame:
    """
    Fetches the upcoming fixtures from the FPL API.
    """
    fixtures = __all_fixtures()
    next_week, teams = __boostrap_from_fpl_api()
    fixtures = fixtures[fixtures["event"] >= next_week]
    fixtures["team_h"] = fixtures["team_h"].map(teams)
    fixtures["team_a"] = fixtures["team_a"].map(teams)
    return fixtures[["event", "team_h", "team_a"]]
