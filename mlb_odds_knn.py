import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

mlb_df = pd.read_csv('/Users/Leander/Desktop/Projects/mlbs_odds/oddsDataMLB.csv')
print(mlb_df.info())