# Load libraries
# data oversampling algorithms on the phoneme imbalanced dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y

    # evaluate a model
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the model evaluation metric
    metric = make_scorer(geometric_mean_score)
    #metric='recall'
    # evaluate model
    scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
    return scores
# define oversampling models to test
def get_models():
    models, names = list(), list()
    # RandomOverSampler
    models.append(RandomOverSampler())
    names.append('ROS')
    # SMOTE
    models.append(SMOTE())
    names.append('SMOTE')
    # BorderlineSMOTE
    models.append(BorderlineSMOTE())
    names.append('BLSMOTE')
    # SVMSMOTE
    models.append(SVMSMOTE())
    names.append('SVMSMOTE')
    # ADASYN
    models.append(ADASYN())
    names.append('ADASYN')
    return models, names
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define models
models, names = get_models()
results = list()
# evaluate each model
for i in range(len(models)):
    # define the model
    model = RandomForestClassifier()
    # define the pipeline steps
    steps = [('s', MinMaxScaler()), ('o', models[i]), ('m', model)]
    # define the pipeline
    pipeline = Pipeline(steps=steps)
    # evaluate the model and store results
    scores = evaluate_model(X, y, pipeline)
    results.append(scores)
    # summarize and store
    print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = RandomForestClassifier(n_estimators=100)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = RandomForestClassifier(n_estimators=100)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = LogisticRegression(solver='liblinear')
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = LogisticRegression(solver='lbfgs')
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = SVC(gamma='auto',C=5)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = SVC(gamma='auto',C=1)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = DecisionTreeClassifier(max_depth=10,min_samples_split=2,min_samples_leaf=2)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = DecisionTreeClassifier(max_depth=10,min_samples_split=2,min_samples_leaf=2)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = SVC(gamma='auto')
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = SVC(gamma='auto')
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = XGBClassifier(scale_pos_weight=99)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = XGBClassifier(scale_pos_weight=99)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = GaussianNB(var_smoothing=0.000001)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = GaussianNB(var_smoothing=0.000001)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = KNeighborsClassifier(n_neighbors=5)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = KNeighborsClassifier(n_neighbors=5)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = LinearDiscriminantAnalysis(solver='lsqr')
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SVMSMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost.sklearn import XGBClassifier

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = LinearDiscriminantAnalysis(solver='lsqr')
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SVMSMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
print(results.mean(), results.std())

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    data = read_csv(full_path, header=None)
    # retrieve numpy array
    data = data.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    return X, y
# define the location of the dataset
full_path = 'dengue_dataset_7_tuning.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = RandomForestClassifier()
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', SMOTE()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
# evaluate on some nasal cases (known class 0)
print('nonDB:')
data = [[35,38,0,0,0,0,1,0,0,0],
    [19,37,0,0,0,1,1,0,0,0],
    [8,37,1,3,0,0,0,0,0,0]]
for row in data:
    # make prediction
    yhat = pipeline.predict([row])
    # get the label
    label = yhat[0]
    # summarize
    print('>Predicted=%d (expected 0)' % (label))
# evaluate on some oral cases (known class 1)
print('DB:')
data = [[0.8,36,1,4,0,0,0,0,1,0],
    [3,38.9,1,5,0,0,0,0,1,0],
    [3.6,37,1,3,0,1,0,0,0,0],
    [4,40.6,1,3,0,0,0,0,1,0],
    [5,39.2,1,7,0,1,0,0,1,0],
    [5,38.9,1,2,0,0,1,0,1,0],
    [5,37,1,3,0,1,0,0,1,0],
    [6,36.2,1,3,0,1,0,0,1,0],
    [6,37.1,1,4,0,0,0,0,1,0],
    [6,36.7,1,3,0,1,1,0,1,0],
    [6,38.2,1,4,0,1,0,0,1,0],
    [6,38.4,1,4,0,1,0,0,1,0],
    [7,38.9,1,5,1,0,0,0,0,0],
    [7,36,1,4,0,0,0,0,1,0],
    [8,37.5,1,4,1,0,0,0,0,0],
    [8,36.7,1,1,0,0,0,0,1,0],
    [8,37,1,6,0,1,1,0,1,0],
    [8,36.5,1,5,0,0,0,0,1,0],
    [8,37.1,1,6,0,1,0,0,1,0],
    [8,37,1,4,0,1,0,0,1,0],
    [9,37.5,1,3,0,1,0,0,1,0],
    [9,37.2,1,6,0,1,0,0,1,0],
    [10,39.7,1,4,0,1,0,0,1,0],
    [10,36,1,3,0,1,0,0,1,0],
    [10,37.1,1,5,0,0,0,0,1,0],
    [10,38.1,1,2,0,1,0,1,1,0],
    [10,36.5,1,4,0,1,1,0,1,0],
    [10,36.9,1,4,0,1,0,0,1,0],
    [11,37.6,1,3,0,1,0,0,1,0],
    [11,38.6,1,1,0,1,1,0,1,0],
    [11,38,1,3,0,1,0,0,1,0],
    [12,36.6,1,4,0,1,0,0,1,0],
    [12,39.7,1,1,0,0,1,0,1,0],
    [13,36.5,1,6,1,0,0,0,0,0],
    [13,37,1,4,0,0,0,0,1,0],
    [14,36,0,0,0,1,0,0,0,0],
    [14,36.5,1,7,0,1,0,0,1,0],
    [14,36.7,1,3,0,1,1,0,1,0],
    [14,38.2,1,3,0,1,0,0,0,0],
    [15,38.7,1,14,0,1,0,0,1,0],
    [15,38.5,1,7,0,0,0,0,1,0],
    [16,36.8,1,4,0,1,0,0,1,0],
    [16,37.4,1,4,0,1,1,0,1,0],
    [16,38,1,4,0,1,0,0,1,0],
    [16,37.5,1,1,0,1,0,0,1,0],
    [16,38,1,4,0,1,1,0,0,0],
    [16,38.5,1,6,0,1,1,0,1,0],
    [17,38,1,3,0,1,1,0,1,0],
    [17,37.6,1,8,0,0,0,0,1,0],
    [17,39.5,1,3,0,1,1,0,1,0],
    [18,38,1,2,0,0,0,0,1,0],
    [18,37.6,1,3,0,1,0,0,1,0],
    [19,37,1,3,0,1,0,0,0,0],
    [19,37.5,1,3,0,0,0,0,1,0],
    [19,37.9,1,4,0,1,1,0,0,0],
    [19,37.5,1,2,0,1,1,0,1,0],
    [21,37,1,4,1,1,0,0,1,0],
    [22,39.2,1,3,0,1,1,0,1,0],
    [22,36.5,1,3,0,1,1,0,0,0],
    [23,37.2,1,2,0,1,0,0,1,0],
    [24,36.2,1,3,0,0,1,0,1,0],
    [24,36,1,4,0,0,1,0,1,0],
    [24,37.6,1,3,0,1,1,0,1,0],
    [24,36.8,1,3,1,1,1,1,1,0],
    [25,37.6,1,6,1,1,1,0,1,0],
    [25,37,1,3,1,0,1,0,1,0],
    [27,36.8,1,3,0,0,1,0,1,0],
    [30,37.6,1,3,0,1,1,1,1,0],
    [34,36.6,1,4,1,1,0,0,1,0],
    [41,37.6,1,3,0,1,1,0,1,0],
    [47,39,1,3,0,1,1,0,1,0],
    [47,36,0,0,0,0,1,1,0,0],
    [53,36,0,0,0,0,0,0,1,1],
    [55,39,1,7,1,1,1,0,0,0],
    [57,36.5,1,4,0,1,1,0,0,0],
    [61,38,1,4,1,1,1,1,1,0],
    [61,36.7,1,3,0,1,1,0,1,0],
    [67,38.6,1,5,0,1,0,0,1,0],
    [78,38,1,3,1,1,1,0,0,0]]
for row in data:
    # make prediction
    yhat = pipeline.predict([row])
    # get the label
    label = yhat[0]
    # summarize
    print('>Predicted=%d (expected 1)' % (label))