#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
# import warnings
# warnings.filterwarnings("ignore")

#!pip install category_encoders
import category_encoders as ce
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_parquet('X_train.parquet')
test = pd.read_parquet('X_test.parquet')
y = pd.read_parquet('y_train.parquet')

pd.set_option('display.max_columns', None)
df = df.dropna()
num_features = [['ЭКСГАУСТЕР 4. ТОК РОТОРА 1', 'ЭКСГАУСТЕР 4. ТОК РОТОРА2',
       'ЭКСГАУСТЕР 4. ТОК СТАТОРА', 'ЭКСГАУСТЕР 4. ДАВЛЕНИЕ МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 4. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 4. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 4. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 4. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 4. ТЕМПЕРАТУРА МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 4. ТЕМПЕРАТУРА МАСЛА В МАСЛОБЛОКЕ',
       'ЭКСГАУСТЕР 4. ВИБРАЦИЯ НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 4. ВИБРАЦИЯ НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 4. ВИБРАЦИЯ НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 4. ВИБРАЦИЯ НА ОПОРЕ 3. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 4. ВИБРАЦИЯ НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 4. ВИБРАЦИЯ НА ОПОРЕ 4. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 5. ТОК РОТОРА 1', 'ЭКСГАУСТЕР 5. ТОК РОТОРА 2',
       'ЭКСГАУСТЕР 5. ТОК СТАТОРА', 'ЭКСГАУСТЕР 5. ДАВЛЕНИЕ МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 5. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 5. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 5. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 5. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 5. ТЕМПЕРАТУРА МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 5. ТЕМПЕРАТУРА МАСЛА В МАСЛОБЛОКЕ',
       'ЭКСГАУСТЕР 5. ВИБРАЦИЯ НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 5. ВИБРАЦИЯ НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 5. ВИБРАЦИЯ НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 5. ВИБРАЦИЯ НА ОПОРЕ 3. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 5. ВИБРАЦИЯ НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 5. ВИБРАЦИЯ НА ОПОРЕ 4. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 6. ТОК РОТОРА 1', 'ЭКСГАУСТЕР 6. ТОК РОТОРА 2',
       'ЭКСГАУСТЕР 6. ТОК СТАТОРА', 'ЭКСГАУСТЕР 6. ДАВЛЕНИЕ МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 6. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 6. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 6. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 6. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 6. ТЕМПЕРАТУРА МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 6. ТЕМПЕРАТУРА МАСЛА В МАСЛОБЛОКЕ',
       'ЭКСГАУСТЕР 6. ВИБРАЦИЯ НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 6. ВИБРАЦИЯ НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 6. ВИБРАЦИЯ НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 6. ВИБРАЦИЯ НА ОПОРЕ 3. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 6. ВИБРАЦИЯ НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 6. ВИБРАЦИЯ НА ОПОРЕ 4. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 7. ТОК РОТОРА 1', 'ЭКСГАУСТЕР 7. ТОК РОТОРА 2',
       'ЭКСГАУСТЕР 7. ТОК СТАТОРА', 'ЭКСГАУСТЕР 7. ДАВЛЕНИЕ МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 7. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 7. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 7. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 7. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 7. ТЕМПЕРАТУРА МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 7. ТЕМПЕРАТУРА МАСЛА В МАСЛОБЛОКЕ',
       'ЭКСГАУСТЕР 7. ВИБРАЦИЯ НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 7. ВИБРАЦИЯ НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 7. ВИБРАЦИЯ НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 7. ВИБРАЦИЯ НА ОПОРЕ 3. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 7. ВИБРАЦИЯ НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 7. ВИБРАЦИЯ НА ОПОРЕ 4. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 8. ТОК РОТОРА 1', 'ЭКСГАУСТЕР 8. ТОК РОТОРА 2',
       'ЭКСГАУСТЕР 8. ТОК СТАТОРА', 'ЭКСГАУСТЕР 8. ДАВЛЕНИЕ МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 8. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 8. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 8. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 8. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 8. ТЕМПЕРАТУРА МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 8. ТЕМПЕРАТУРА МАСЛА В МАСЛОБЛОКЕ',
       'ЭКСГАУСТЕР 8. ВИБРАЦИЯ НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 8. ВИБРАЦИЯ НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 8. ВИБРАЦИЯ НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 8. ВИБРАЦИЯ НА ОПОРЕ 3. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 8. ВИБРАЦИЯ НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 8. ВИБРАЦИЯ НА ОПОРЕ 4. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 9. ТОК РОТОРА 1', 'ЭКСГАУСТЕР 9. ТОК РОТОРА 2',
       'ЭКСГАУСТЕР 9. ТОК СТАТОРА', 'ЭКСГАУСТЕР 9. ДАВЛЕНИЕ МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 9. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 9. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 9. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 9. ТЕМПЕРАТУРА ПОДШИПНИКА НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 9. ТЕМПЕРАТУРА МАСЛА В СИСТЕМЕ',
       'ЭКСГАУСТЕР 9. ТЕМПЕРАТУРА МАСЛА В МАСЛОБЛОКЕ',
       'ЭКСГАУСТЕР 9. ВИБРАЦИЯ НА ОПОРЕ 1',
       'ЭКСГАУСТЕР 9. ВИБРАЦИЯ НА ОПОРЕ 2',
       'ЭКСГАУСТЕР 9. ВИБРАЦИЯ НА ОПОРЕ 3',
       'ЭКСГАУСТЕР 9. ВИБРАЦИЯ НА ОПОРЕ 3. ПРОДОЛЬНАЯ.',
       'ЭКСГАУСТЕР 9. ВИБРАЦИЯ НА ОПОРЕ 4',
       'ЭКСГАУСТЕР 9. ВИБРАЦИЯ НА ОПОРЕ 4. ПРОДОЛЬНАЯ.']


num_features_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=False)),
    ('scaler', RobustScaler())
])

CT = ColumnTransformer([ 
        ("num", num_features_transformer, num_features),
        ])

display(CT)

res_ct = CT.fit_transform(df)

X_trans = pd.DataFrame(res_ct,  index = df.index)

                
window = 10800# 3 часа
for column in X_trans.columns:
    X_trans[str(column) + '_rolling_mean'] = X_trans[column].rolling(window).mean()
    X_trans[str(column) + '_rolling_std'] = X_trans[column].rolling(window).std()
X_trans = X_trans.dropna()

num_rows_diff = len(df) - len(X_trans)
y = y[num_rows_diff:]
                
# Initialize a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42)
X_train.columns = X_trans.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

