"""
#(prac7)-----
#AIM: Implement AdaBoost(Adaptive Boosting) learning algorithm.
"""
'''AdaBoost also called Adaptive Boosting is a technique in Machine
Learning used as an Ensemble Method. The most common algorithm used
with AdaBoost is decision trees with one level that means with
Decision trees with only 1 split. These trees are also called Decision Stumps.

'''
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 30
#kfold makes trees with split number.
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
#n_estimators : This is the number of trees you want to build before predictions.
#Higher number of trees give you better voting optionsand perfomance performance 
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
#cross_val_score method is used to calculate the accuracy of model sliced into x, y
#cross validator cv  is optional cv=kfold
results = model_selection.cross_val_score(model, X, Y)
print(results.mean())


"""
OUTPUT will come after seconds:

0.755287171613133
"""
