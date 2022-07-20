import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV

def get_files_list(home_dir):
    files = []

    for file in os.listdir(home_dir):
        if file.endswith(".csv"):
            files.append(os.path.join(home_dir, file))
    
    return files

def get_data_from_csv(filename):
    word_data = np.loadtxt(filename, delimiter=",")
    X = word_data[:,:-1]
    y = word_data[:,-1:]
    return X,y

# get filename
home_dir = './train/'
filename = get_files_list(home_dir)
print(filename)

# get data
print(filename[0])
X,y = get_data_from_csv(filename[0])
y_train = y.reshape(-1,)

param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [10,1,0.1,0.01,0.001,0.0001,'scale', 'auto'],
              'kernel': ['poly', 'rbf', 'linear', 'sigmoid']
            }
clf = svm.SVC(degree=2)
gs = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
gs.fit(X, y_train)
print('Best acccuracy: %.6f' % gs.best_score_)
print(gs.best_params_)