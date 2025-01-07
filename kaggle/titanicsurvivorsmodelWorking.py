import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)]

train_pd = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data = train_pd.dropna()
# print(train_data)

# if test_data.isnull().values.any():
#     print("There are NaN values in the dataset.")
# else:
#     print("No NaN values in the dataset.")

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
# print(test_data)

submission_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
print(submission_data)

women = train_data.loc[train_data.Sex == 'female']['Survived']
#sum(women) gets all the sums of the women array [1,1,0,1,1] = 4
#len(women) gets the length of the women array = 314 
#...so sum = 233; therefore 314 - 233 = 81 women didn't survive. rate =  0.25796178343949044
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women) #0.7420382165605095

train_data.describe()

men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men) #0.18890814558058924

from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier, HistGradientBoostingClassifier #RandomForestRegressor, 
from sklearn.metrics import mean_absolute_error

# from sklearn.experimental import enable_iterative_imputer #needed import for IterativeImputer
# from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import xgboost as xgb
# from sklearn.model_selection import train_test_split


# y = train_data.Survived
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
# X = pd.get_dummies[train_data[features]]
X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies[test_data[features]]
X_test_pd = pd.get_dummies(test_data[features])
X_test = X_test_pd#.dropna()
# 
# imputer = SimpleImputer(strategy='most_frequent') #most_frequent = 192 #constant = 191
# imputer = IterativeImputer()
# X_test = pd.DataFrame(imputer.fit_transform(X_test_pd), columns=X_test_pd.columns)

# train_X, train_y, val_X, val_y = train_test_split(X, y, random_state=1)

val_y = submission_data.Survived.values
# print(val_y)

if X_test.isnull().values.any():
    print("There are NaN values in the dataset.")
else:
    print("No NaN values in the dataset.")
    
# model = xgb.XGBClassifier(max_depth=10, random_state=0) #216
# #
# model.fit(X, y)
# classifier_predictions = model.predict(X_test)

# RandomForestClassifier(random_state=0, bootstrap=False, max_leaf_nodes=100, n_estimators = 100, max_depth=10) 
classifier_model = RandomForestRegressor(random_state=1, warm_start=False, max_leaf_nodes=100, n_estimators = 100, max_depth = 5, bootstrap=False) 
#m_d=5: 256 #213, bootstap=False:256
# classifier_model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 0 ) #v1 213 #192
# classifier_model = HistGradientBoostingClassifier( max_leaf_nodes=200, max_depth = 10, random_state=0) #235
classifier_model.fit(X, y)
classifier_predictions = classifier_model.predict(X_test)

print(classifier_predictions)

prediction = []
for i in classifier_predictions:
    prediction.append( round(i) )

print(prediction)

# mmm = mean_absolute_error(classifier_predictions, val_y)
# print(" Mean Absolute Error:  %d" %(mmm))
print("predict: %d  \t\t val_y:  %d" %(len(classifier_predictions), len(val_y) ))

true_count = 0
false_count = 0
for i in prediction:
    idx = 0
    if i == val_y[idx]:
        true_count = true_count + 1
        idx = idx + 1
    else:
        false_count = false_count + 1
print("true_count: %d  \t\t false_count:  %d" %(true_count, false_count))
    
        
    
# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': prediction })

# output.to_csv('submission.csv', index=False )
# # output.to_csv('submission.csv', index=false)
# print("Your submission was successfully saved!")

# nodes = [5, 25, 50, 100, 250, 500, 5000 ]

# def get_mae( model_type, max_leaf_nodes, state, train_X, train_y, val_X, val_y):
#     if model_type == 'regressor':
#         model = RandomForestRegressor(max_tree_deps,random_state=state)
#         model.fit(train_X, train_y)
#         prediction = model.predict(val_X)
#         mae = mean_absolute_error(prediction, val_y)
#         return mae
#     else:
#         model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = state )

# nodes = [5, 25, 50, 100, 250, 500, 5000 ]
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# print(train_X, "---- train_y", train_y, "---- val_X", val_X, "----val_y", val_y)
# for max_leaf_nodes in nodes:
#     mae = get_mae( max_leaf_nodes, 1, train_X, train_y, val_X, val_y)
#     print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae))

# submission_data = pd.read_csv('/kaggle/working/submission.csv')
# submission_data.head()
# input_submission_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
