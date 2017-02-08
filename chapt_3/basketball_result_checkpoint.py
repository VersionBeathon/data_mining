#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/2/7 13:06
# @Author :Kira
# @Software：PyCharm
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


clf = DecisionTreeClassifier(random_state=14)
data_filename = ' NBA_2013_2014.csv'
standings_filename = 'expend_standing.csv'
standings = pd.read_csv(standings_filename, skiprows=[0, ])
standings.columns = ["Rk", "Team", "Home", "Overall", "Road", "E", "W", "A", "C", "SE", "NW", "P", "SW", "Pre", "Post",
                     "<=3", ">=10", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]

dataset = pd.read_csv(data_filename, parse_dates=[0], skiprows=[0, ])
dataset.columns = ["Date", "Start", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "Score Type", "OT?", "Notes"]
dataset["homewin"] = dataset["VisitorPts"] < dataset["HomePts"]
y_true = dataset["homewin"].values
dataset["HomeLastWin"] = False
dataset["VisitorLastWin"] = False
won_last = defaultdict(int)


for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    dataset.ix[index] = row
    won_last[home_team] = row["homewin"]
    won_last[visitor_team] = not row["homewin"]

X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values
scores = cross_val_score(clf, X_previouswins, y_true, scoring="accuracy")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

dataset["HomeTeamRanksHigher"] = 0
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    if home_team == "New Orleans Pelicans":
        home_team = "New Orleans Hornets"
    elif visitor_team == "New Orleans Pelicans":
        visitor_team = "New Orleans Hornets"
    home_rank = standings[standings["Team"] == home_team]["Rk"].values[0]
    visitor_rank = standings[standings["Team"] == visitor_team]["Rk"].values[0]
    row["HomeTeamRanksHigher"] = int(home_rank > visitor_rank)
    dataset.ix[index] = row

X_homehigher = dataset[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
scores = cross_val_score(clf, X_homehigher, y_true, scoring="accuracy")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

last_match_winner = defaultdict(int)
dataset["HomeTeamWonLast"] = 0
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    teams = tuple(sorted([home_team, visitor_team]))
    row["HomeTeamWonLast"] = 1 if last_match_winner[teams] == row["Home Team"] else 0
    dataset.ix[index] = row
    winner = row["Home Team"] if row["homewin"] else row["Visitor Team"]
    last_match_winner[teams] = winner
X_lastwinner = dataset[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher", "HomeTeamWonLast"]].values
scores = cross_val_score(clf, X_lastwinner, y_true, scoring="accuracy")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

encoding = LabelEncoder()
encoding.fit(dataset["Home Team"].values)
home_team = encoding.transform(dataset["Home Team"].values)
visitor_team = encoding.transform(dataset["Visitor Team"].values)
X_teams = np.vstack([home_team, visitor_team]).T
onehot = OneHotEncoder()
X_teams = onehot.fit_transform(X_teams).todense()
scores = cross_val_score(clf, X_teams, y_true, scoring="accuracy")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

clt = RandomForestClassifier(random_state=14)
scores = cross_val_score(clt, X_teams, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

X_all = np.hstack([X_homehigher, X_teams])
scores = cross_val_score(clt, X_all, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

parameter_space = {
                   "max_features": [2, 10, 'auto'],
                   "n_estimators": [100,],
                   "criterion": ["gini", "entropy"],
                   "min_samples_leaf": [2, 4, 6],
                   }
grid = GridSearchCV(clt, parameter_space)
grid.fit(X_all, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))
