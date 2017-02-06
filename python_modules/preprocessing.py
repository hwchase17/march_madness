#!/usr/bin/env python

""" preprocessing.py: 
This is a module to perform preprocessing on data used in the 2017 March Madness Challenge on www.kaggle.com.
Some data is from the data given in the competition and other is scraped form www.kenpom.com.
"""

# Import packages
import pandas as pd
import numpy as np
import random

class data:
    """
    Defines a data class for training march madness results.
    """
    
    def __init__(self):
        """
        Initialize the class using hardcoded folder destinations.
        """
        self.data = pd.DataFrame.from_csv("data/kenpom_team_ratings.csv")
        self.seeds = pd.DataFrame.from_csv("data/TourneySeeds.csv")
        self.teams = pd.DataFrame.from_csv("data/Teams.csv")
        self.tourney_data = pd.DataFrame.from_csv("data/TourneyCompactResults.csv")
        self.spellings = pd.DataFrame.from_csv("data/TeamSpellings.csv")
        self.all_features = self.data.columns
        
        # Initialize lists for kenpom teams and kaggle teams (to check that they match)
        self.years = range(2002,2017)
        self.teams_from_kenpom = [[] for _ in self.years]
        self.teams_from_kaggle = [[] for _ in self.years]
    
    def import_data(self):
        """    
        Imports data into the initialized data variables.
        """

        # Define names with team IDs that have not yet been defined in 'TeamSpellings.csv'
        names_to_add = {'mississippi valley st.':'1290',
                        'arkansas pine bluff':'1115',
                        'arkansas little rock':'1114',
                        'louisiana lafayette':'1418',
                        'cal st. bakersfield':'1167',
                        'illinois chicago':'1227',
                        'texas a&m corpus chris':'1394'}

        # Add names from kenpoms website to spellings
        for name_to_add in names_to_add:
            self.spellings.ix[name_to_add,:] = names_to_add[name_to_add]

        # Take seed out from name
        a = self.data['Team'].copy()
        a.sort_values(axis=0)
        a.unique()
        b = pd.Series()
        for i, a_ in enumerate(a):
            new = a_.split(" ")
            if new[-1].isdigit():
                new = " ".join(new[:-1])
                b.set_value(i, new)
            else:
                b.set_value(i, a_)
        b = sorted(b.unique())

        # Save teams from kenpom
        for i, year in enumerate(self.years):
            year_seeds = list(self.seeds.loc[str(year)]['Team'])
            seeds_used = []
            for b_ in b:
                try:
                    team_id = self.spellings.loc[b_.lower()]
                    if int(team_id) in year_seeds and int(team_id) not in seeds_used:
                        self.teams_from_kenpom[i].append([b_, int(team_id)])
                        seeds_used.append(int(team_id))
                except KeyError:
                    pass

        # Save teams from kaggle
        for i, year in enumerate(self.years):
            year_seeds = list(self.seeds.loc[str(year)]['Team'])
            for seed in year_seeds:
                index = seed - 1101
                name = self.teams.iloc[index]
                self.teams_from_kaggle[i].append([name[0], int(seed)])

        # Sort teams according to team ID
        for i in xrange(len(self.years)):
            self.teams_from_kenpom[i].sort(key=lambda x: x[1])
            self.teams_from_kaggle[i].sort(key=lambda x: x[1])

        return None

    def compare_kaggle_kenpom_teams(self):
        """
        Compare kaggle names to kenpom names to ensure that all teams are correctly saved
        Print this when adding new teams to ensure that teams are added correctly, the lengths should always match
        """
        for i in xrange(self.years):
            print len(self.teams_from_kenpom[i])
            print len(self.teams_from_kaggle[i])
        
        return None
    
    def XY(self, features, cv_years, randomize=True):
        """
        Get X and Y matrices from the data and cv_indices according to the years chosen as cross-validation data.
        Note that X and Y includes the cross-validation data.
        
        input:
            features: features from the data to be included in the X matrix (list of strings)
            cv_years: years to be used for cross-validation (list of ints)
            randomize: True if X and Y columns are to be randomized (i.e., which team is left/right) (boolean variable)
        """
        
        X = pd.DataFrame()
        Y = []
        cv_indices = []
        
        features_to_include_both_teams = []
        for feature in features:
            features_to_include_both_teams.append(feature + '_x')
            features_to_include_both_teams.append(feature + '_y')
            
        index = 0

        for i, year in enumerate(self.years):
            year_teams = pd.DataFrame(self.teams_from_kenpom[i])
            year_teams.columns = ['Team', 'Team_ID']
            year_data = self.data[self.data['Year'] == int(year)]
            year_data = pd.merge(year_data, year_teams, on=['Team'])
            year_results = self.tourney_data.loc[str(year)]

            year_data = year_data.rename(columns={'Team_ID':'Wteam'})
            year_results = pd.merge(year_results, year_data, on='Wteam')
            year_data = year_data.rename(columns={'Wteam':'Lteam'})
            year_results = pd.merge(year_results, year_data, on='Lteam', how='outer')
            year_results = year_results[year_results['Wteam'] == year_results['Wteam']]

            # Temporary, should consider ways to deal with missing data
            year_results = year_results.dropna()

            y = year_results['Wscore'] - year_results['Lscore']
            year_results = year_results[features_to_include_both_teams]

            X = X.append(year_results, ignore_index=True)
            Y += list(y)

            if year in cv_years:
                indices_for_cv = range(index,index + len(y))
                cv_indices += indices_for_cv

            index += len(y)
        
        train_indices = list(set(range(977)) - set(cv_indices))
        X_cv = X.loc[cv_indices,:]
        X = X.loc[train_indices,:]
        Y_cv = np.array(Y)[cv_indices]
        Y = list(np.array(Y)[train_indices])   
        
        if randomize:
            # Randomize the order of the teams
            switch_array = np.array([0] * (X.shape[0]/2) + [1] * (X.shape[0] - X.shape[0]/2))
            np.random.shuffle(switch_array)
            X_random = pd.DataFrame()
            Y_random = [-y if x == 0 else y for x,y in zip(switch_array, Y)]
            for feature in features:
                X_random[feature + '_x'] = X[feature + '_x'] * switch_array + X[feature + '_y'] * (1 - switch_array)
                X_random[feature + '_y'] = X[feature + '_x'] * (1 - switch_array) + X[feature + '_y'] * switch_array
                
            X = X_random
            Y = np.array(Y_random)
            
        return X, X_cv, Y, Y_cv, cv_indices

