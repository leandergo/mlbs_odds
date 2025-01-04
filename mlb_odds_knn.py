import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pdb

mlb_df = pd.read_csv('/Users/Leander/Desktop/Projects/mlbs_odds/oddsDataMLB.csv')
mlb_df = mlb_df.drop(['parkName', 'oppMoneyLine', 'oppRunLine', 'oppRunLineOdds', 'runLineOdds', 'runLine', 'projectedRuns', 'runDif', 'overOdds', 'underOdds'], axis=1)
mlb_df['date'] = pd.to_datetime(mlb_df['date'])
mlb_df = mlb_df.sort_values(by='date', ascending=False)
mlb_df['under'] = np.where(mlb_df['total'] < mlb_df['totalRuns'], 1, 0)
mlb_df['over'] = np.where(mlb_df['total'] > mlb_df['totalRuns'], 1, 0)
mlb_df['push'] = np.where(mlb_df['total'] == mlb_df['totalRuns'], 1, 0)

def odds_to_prob(odd):
    return abs(odd) / (abs(odd) + 100)


mlb_df['percent'] = mlb_df['moneyLine'].apply(odds_to_prob).round(3)
mlb_df = mlb_df.drop(['moneyLine'], axis=1)
mlb_df['total_standard'] = ((mlb_df['total'] - mlb_df['total'].mean()) / mlb_df['total'].std()).round(3)


# Splitting the data into training and testing sets
features = ['total_standard', 'percent']
target = ['under']

for seasons in [2018, 2019, 2021]:
    print(f'\n Results for {seasons} season:')

    train_df = mlb_df.query('season < @seasons')
    test_df = mlb_df.query('season == @seasons and totalRuns != total')

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    model = KNeighborsClassifier(n_neighbors=101)

    clf = model.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_true = y_test

    # pdb.set_trace()

    print(f'accruacy = {accuracy_score(y_true, y_pred):.1%}')
    print(classification_report(y_true, y_pred, target_names=['over', 'under']))

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['over', 'under'])
    display.plot()
    plt.grid(False)
    plt.show()
