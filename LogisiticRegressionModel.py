import pandas as pd
import numpy as np
import sklearn as sk

from sportsreference.nfl.teams import Teams
from sportsreference.nfl.boxscore import Boxscores, Boxscore

from datetime import datetime


# year, month, day


class LogisticRegressionModel:

    def get_nfl_schedule(self):
        week_list = list(range(1, 18))
        nfl_schedule = pd.DataFrame()
        year = 2019

        for week in week_list:
            week_str = str(week)
            year_str = str(year)
            date = week_str + "-" + year_str
            week_results = Boxscores(week, year)

            week_results_df = pd.DataFrame()
            for game in week_results.games[date]:
                game_results = pd.DataFrame(game, index=[0])[
                    ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'winning_name', 'winning_abbr']]
                game_results['week'] = week
                week_results_df = pd.concat([week_results_df, game_results])

            nfl_schedule = pd.concat([nfl_schedule, week_results_df]).reset_index().drop(columns='index')

        return nfl_schedule

    def game_stats_cleanup(self, game_df, game_stats):
        try:
            away_team_df = game_df[['away_name', 'away_abbr', 'away_score']].rename(
                columns={'away_name': 'team_name', 'away_abbr': 'team_abbr', 'away_score': 'score'})
            home_team_df = game_df[['home_name', 'home_abbr', 'home_score']].rename(
                columns={'home_name': 'team_name', 'home_abbr': 'team_abbr', 'home_score': 'score'})
            try:
                if game_df.loc[0, 'away_score'] > game_df.loc[0, 'home_score']:
                    away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won': [1], 'game_lost': [0]}),
                                            left_index=True, right_index=True)
                    home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won': [0], 'game_lost': [1]}),
                                            left_index=True, right_index=True)
                elif game_df.loc[0, 'away_score'] < game_df.loc[0, 'home_score']:
                    away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won': [0], 'game_lost': [1]}),
                                            left_index=True, right_index=True)
                    home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won': [1], 'game_lost': [0]}),
                                            left_index=True, right_index=True)
                else:
                    away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won': [0], 'game_lost': [0]}),
                                            left_index=True, right_index=True)
                    home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won': [0], 'game_lost': [0]}),
                                            left_index=True, right_index=True)
            except TypeError:
                away_team_df = pd.merge(away_team_df, pd.DataFrame({'game_won': [np.nan], 'game_lost': [np.nan]}),
                                        left_index=True, right_index=True)
                home_team_df = pd.merge(home_team_df, pd.DataFrame({'game_won': [np.nan], 'game_lost': [np.nan]}),
                                        left_index=True, right_index=True)

            away_stats_df = game_stats.dataframe[['away_first_downs', 'away_fourth_down_attempts',
                                                  'away_fourth_down_conversions', 'away_fumbles', 'away_fumbles_lost',
                                                  'away_interceptions', 'away_net_pass_yards', 'away_pass_attempts',
                                                  'away_pass_completions', 'away_pass_touchdowns', 'away_pass_yards',
                                                  'away_penalties', 'away_points', 'away_rush_attempts',
                                                  'away_rush_touchdowns', 'away_rush_yards', 'away_third_down_attempts',
                                                  'away_third_down_conversions', 'away_time_of_possession',
                                                  'away_times_sacked', 'away_total_yards', 'away_turnovers',
                                                  'away_yards_from_penalties',
                                                  'away_yards_lost_from_sacks']].reset_index().drop(
                columns='index').rename(columns={
                'away_first_downs': 'first_downs', 'away_fourth_down_attempts': 'fourth_down_attempts',
                'away_fourth_down_conversions': 'fourth_down_conversions', 'away_fumbles': 'fumbles',
                'away_fumbles_lost': 'fumbles_lost',
                'away_interceptions': 'interceptions', 'away_net_pass_yards': 'net_pass_yards',
                'away_pass_attempts': 'pass_attempts',
                'away_pass_completions': 'pass_completions', 'away_pass_touchdowns': 'pass_touchdowns',
                'away_pass_yards': 'pass_yards',
                'away_penalties': 'penalties', 'away_points': 'points', 'away_rush_attempts': 'rush_attempts',
                'away_rush_touchdowns': 'rush_touchdowns', 'away_rush_yards': 'rush_yards',
                'away_third_down_attempts': 'third_down_attempts',
                'away_third_down_conversions': 'third_down_conversions',
                'away_time_of_possession': 'time_of_possession',
                'away_times_sacked': 'times_sacked', 'away_total_yards': 'total_yards', 'away_turnovers': 'turnovers',
                'away_yards_from_penalties': 'yards_from_penalties',
                'away_yards_lost_from_sacks': 'yards_lost_from_sacks'})

            home_stats_df = game_stats.dataframe[['home_first_downs', 'home_fourth_down_attempts',
                                                  'home_fourth_down_conversions', 'home_fumbles', 'home_fumbles_lost',
                                                  'home_interceptions', 'home_net_pass_yards', 'home_pass_attempts',
                                                  'home_pass_completions', 'home_pass_touchdowns', 'home_pass_yards',
                                                  'home_penalties', 'home_points', 'home_rush_attempts',
                                                  'home_rush_touchdowns', 'home_rush_yards', 'home_third_down_attempts',
                                                  'home_third_down_conversions', 'home_time_of_possession',
                                                  'home_times_sacked', 'home_total_yards', 'home_turnovers',
                                                  'home_yards_from_penalties',
                                                  'home_yards_lost_from_sacks']].reset_index().drop(
                columns='index').rename(columns={
                'home_first_downs': 'first_downs', 'home_fourth_down_attempts': 'fourth_down_attempts',
                'home_fourth_down_conversions': 'fourth_down_conversions', 'home_fumbles': 'fumbles',
                'home_fumbles_lost': 'fumbles_lost',
                'home_interceptions': 'interceptions', 'home_net_pass_yards': 'net_pass_yards',
                'home_pass_attempts': 'pass_attempts',
                'home_pass_completions': 'pass_completions', 'home_pass_touchdowns': 'pass_touchdowns',
                'home_pass_yards': 'pass_yards',
                'home_penalties': 'penalties', 'home_points': 'points', 'home_rush_attempts': 'rush_attempts',
                'home_rush_touchdowns': 'rush_touchdowns', 'home_rush_yards': 'rush_yards',
                'home_third_down_attempts': 'third_down_attempts',
                'home_third_down_conversions': 'third_down_conversions',
                'home_time_of_possession': 'time_of_possession',
                'home_times_sacked': 'times_sacked', 'home_total_yards': 'total_yards', 'home_turnovers': 'turnovers',
                'home_yards_from_penalties': 'yards_from_penalties',
                'home_yards_lost_from_sacks': 'yards_lost_from_sacks'})

            away_team_df = pd.merge(away_team_df, away_stats_df, left_index=True, right_index=True)
            home_team_df = pd.merge(home_team_df, home_stats_df, left_index=True, right_index=True)
            try:
                away_team_df['time_of_possession'] = (int(away_team_df['time_of_possession'].loc[0][0:2]) * 60) + int(
                    away_team_df['time_of_possession'].loc[0][3:5])
                home_team_df['time_of_possession'] = (int(home_team_df['time_of_possession'].loc[0][0:2]) * 60) + int(
                    home_team_df['time_of_possession'].loc[0][3:5])
            except TypeError:
                away_team_df['time_of_possession'] = np.nan
                home_team_df['time_of_possession'] = np.nan
        except TypeError:
            away_team_df = pd.DataFrame()
            home_team_df = pd.DataFrame()
        return away_team_df, home_team_df

    def get_game_stats_for_season(self):
        week_list = list(range(1, 18))
        nfl_game_stats = pd.DataFrame()
        year = 2019

        for week in week_list:
            week_str = str(week)
            year_str = str(year)
            date = week_str + "-" + year_str
            week_results = Boxscores(week, year)

            week_stats_df = pd.DataFrame()
            for game in week_results.games[date]:
                game_id = game['boxscore']
                game_stats = Boxscore(game_id)
                game_results = pd.DataFrame(game, index=[0])

                away_team_stats, home_team_stats = self.game_stats_cleanup(game_results, game_stats)
                away_team_stats['week'] = week
                home_team_stats['week'] = week
                week_stats_df = pd.concat([week_stats_df, away_team_stats])
                week_stats_df = pd.concat([week_stats_df, home_team_stats])

            nfl_game_stats = pd.concat([nfl_game_stats, week_stats_df])

        return nfl_game_stats

if __name__ == '__main__':
    LRM = LogisticRegressionModel()
    LRM.get_nfl_schedule()
