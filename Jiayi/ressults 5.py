Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd

# Important imports for preprocessing, modeling, and evaluation.
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

# Visualization package imports.
import matplotlib.pyplot as plt
import seaborn as sns
SyntaxError: multiple statements found while compiling a single statement
df_original = pd.read_csv("Invistico_Airline.csv")
Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    df_original = pd.read_csv("Invistico_Airline.csv")
NameError: name 'pd' is not defined. Did you mean: 'id'?
import pandas aspd
SyntaxError: incomplete input
import pandas as pd
df_original = pd.read_csv("Invistico_Airline.csv")
Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    df_original = pd.read_csv("Invistico_Airline.csv")
  File "D:\py\python\Lib\site-packages\pandas\io\parsers\readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "D:\py\python\Lib\site-packages\pandas\io\parsers\readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "D:\py\python\Lib\site-packages\pandas\io\parsers\readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "D:\py\python\Lib\site-packages\pandas\io\parsers\readers.py", line 1880, in _make_engine
    self.handles = get_handle(
  File "D:\py\python\Lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'Invistico_Airline.csv'
df_original.head(10)
Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    df_original.head(10)
NameError: name 'df_original' is not defined
df_original = pd.read_csv("C:/Users/Administrator/Desktop/Invistico_Airline.csv")
df_original.head(10)
  satisfaction  ... Arrival Delay in Minutes
0    satisfied  ...                      0.0
1    satisfied  ...                    305.0
2    satisfied  ...                      0.0
3    satisfied  ...                      0.0
4    satisfied  ...                      0.0
5    satisfied  ...                      0.0
6    satisfied  ...                     15.0
7    satisfied  ...                      0.0
8    satisfied  ...                      0.0
9    satisfied  ...                     26.0

[10 rows x 22 columns]
df_original.dtypes
satisfaction                          object
Customer Type                         object
Age                                    int64
Type of Travel                        object
Class                                 object
Flight Distance                        int64
Seat comfort                           int64
Departure/Arrival time convenient      int64
Food and drink                         int64
Gate location                          int64
Inflight wifi service                  int64
Inflight entertainment                 int64
Online support                         int64
Ease of Online booking                 int64
On-board service                       int64
Leg room service                       int64
Baggage handling                       int64
Checkin service                        int64
Cleanliness                            int64
Online boarding                        int64
Departure Delay in Minutes             int64
Arrival Delay in Minutes             float64
dtype: object
df_original["satisfaction"].value_counts()
satisfaction
satisfied       71087
dissatisfied    58793
Name: count, dtype: int64
df_original.isnull().sum()
satisfaction                           0
Customer Type                          0
Age                                    0
Type of Travel                         0
Class                                  0
Flight Distance                        0
Seat comfort                           0
Departure/Arrival time convenient      0
Food and drink                         0
Gate location                          0
Inflight wifi service                  0
Inflight entertainment                 0
Online support                         0
Ease of Online booking                 0
On-board service                       0
Leg room service                       0
Baggage handling                       0
Checkin service                        0
Cleanliness                            0
Online boarding                        0
Departure Delay in Minutes             0
Arrival Delay in Minutes             393
dtype: int64
df_subset = df_original.dropna(axis=0).reset_index(drop=True)
f_subset = df_subset.astype({"Inflight entertainment": float})
df_subset
SyntaxError: multiple statements found while compiling a single statement
df_subset['satisfaction'] = OneHotEncoder(drop='first').\
fit_transform(df_subset[['satisfaction']]).toarray()
df_subset.satisfaction
SyntaxError: multiple statements found while compiling a single statement
from sklearn.preprocessing import OneHotEncoder
Traceback (most recent call last):
  File "<pyshell#14>", line 1, in <module>
    from sklearn.preprocessing import OneHotEncoder
ModuleNotFoundError: No module named 'sklearn'
df_subset['satisfaction'] = OneHotEncoder(drop='first').\
fit_transform(df_subset[['satisfaction']]).toarray()
Traceback (most recent call last):
  File "<pyshell#16>", line 1, in <module>
    df_subset['satisfaction'] = OneHotEncoder(drop='first').\
NameError: name 'OneHotEncoder' is not defined
from sklearn.preprocessing import OneHotEncoder

Traceback (most recent call last):
  File "<pyshell#17>", line 1, in <module>
    from sklearn.preprocessing import OneHotEncoder
ModuleNotFoundError: No module named 'sklearn'
df_original = pd.read_csv(r"C:\Users\Administrator\Desktop\Invistico_Airline.csv")
df_subset.satisfaction
0            satisfied
1            satisfied
2            satisfied
3            satisfied
4            satisfied
              ...     
129482       satisfied
129483    dissatisfied
129484    dissatisfied
129485    dissatisfied
129486    dissatisfied
Name: satisfaction, Length: 129487, dtype: object
df_subset.head(10)
  satisfaction  ... Arrival Delay in Minutes
0    satisfied  ...                      0.0
1    satisfied  ...                    305.0
2    satisfied  ...                      0.0
3    satisfied  ...                      0.0
4    satisfied  ...                      0.0
5    satisfied  ...                      0.0
6    satisfied  ...                     15.0
7    satisfied  ...                      0.0
8    satisfied  ...                      0.0
9    satisfied  ...                     26.0

[10 rows x 22 columns]
X = df_subset[["Inflight entertainment"]]
y = df_subset[["satisfaction"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
Traceback (most recent call last):
  File "<pyshell#23>", line 1, in <module>
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
NameError: name 'train_test_split' is not defined
from sklearn.model_selection import train_test_split
Traceback (most recent call last):
  File "<pyshell#24>", line 1, in <module>
    from sklearn.model_selection import train_test_split
ModuleNotFoundError: No module named 'sklearn'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Traceback (most recent call last):
  File "<pyshell#25>", line 1, in <module>
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
NameError: name 'train_test_split' is not defined
from sklearn.model_selection import train_test_split
Traceback (most recent call last):
  File "<pyshell#26>", line 1, in <module>
    from sklearn.model_selection import train_test_split
ModuleNotFoundError: No module named 'sklearn'
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression().fit(X_train, y_train)
Traceback (most recent call last):
  File "<pyshell#29>", line 1, in <module>
    clf = LogisticRegression().fit(X_train, y_train)
NameError: name 'LogisticRegression' is not defined
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_train, y_train)
clf.coef_
array([[0.99752883]])
clf.intercept_
array([-3.19359054])
sns.regplot(x="Inflight entertainment", y="satisfaction", 
            data=df_subset, logistic=True, ci=None)
Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    sns.regplot(x="Inflight entertainment", y="satisfaction",
NameError: name 'sns' is not defined
import seaborn as snssns.regplot(x="Inflight entertainment", y="satisfaction",
            data=df_subset, logistic=True, ci=None)
Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    sns.regplot(x="Inflight entertainment", y="satisfaction",
NameError: name 'sns' is not defined
                
SyntaxError: invalid syntax
import seaborn as sns

sns.regplot(x="Inflight entertainment", y="satisfaction", 
            data=df_subset, logistic=True, ci=None)
SyntaxError: multiple statements found while compiling a single statement
# 第一步：导入必要的库
import seaborn as sns
import matplotlib.pyplot as plt

# 第二步：绘制图形
sns.regplot(x="Inflight entertainment", y="satisfaction", 
            data=df_subset, logistic=True, ci=None)

# 第三步：显示图形
plt.show()
SyntaxError: multiple statements found while compiling a single statement
import seaborn as sns
import matplotlib.pyplot as plt
SyntaxError: multiple statements found while compiling a single statement
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x="Inflight entertainment", y="satisfaction", data=df_subset, logistic=True, ci=None)
Traceback (most recent call last):
  File "<pyshell#41>", line 1, in <module>
    sns.regplot(x="Inflight entertainment", y="satisfaction", data=df_subset, logistic=True, ci=None)
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 775, in regplot
    plotter.plot(ax, scatter_kws, line_kws)
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 384, in plot
    self.lineplot(ax, line_kws)
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 429, in lineplot
    grid, yhat, err_bands = self.fit_regression(ax)
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 218, in fit_regression
    yhat, yhat_boots = self.fit_statsmodels(grid, GLM,
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 295, in fit_statsmodels
    yhat = reg_func(X, y)
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 289, in reg_func
    yhat = model(_y, _x, **kwargs).fit().predict(grid)
  File "D:\py\python\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py", line 326, in __init__
    super().__init__(endog, exog, missing=missing,
  File "D:\py\python\Lib\site-packages\statsmodels\base\model.py", line 270, in __init__
    super().__init__(endog, exog, **kwargs)
  File "D:\py\python\Lib\site-packages\statsmodels\base\model.py", line 95, in __init__
    self.data = self._handle_data(endog, exog, missing, hasconst,
  File "D:\py\python\Lib\site-packages\statsmodels\base\model.py", line 135, in _handle_data
    data = handle_data(endog, exog, missing, hasconst, **kwargs)
  File "D:\py\python\Lib\site-packages\statsmodels\base\data.py", line 675, in handle_data
    return klass(endog, exog=exog, missing=missing, hasconst=hasconst,
  File "D:\py\python\Lib\site-packages\statsmodels\base\data.py", line 84, in __init__
    self.endog, self.exog = self._convert_endog_exog(endog, exog)
  File "D:\py\python\Lib\site-packages\statsmodels\base\data.py", line 509, in _convert_endog_exog
    raise ValueError("Pandas data cast to numpy dtype of object. "
ValueError: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).
plt.show()
                     
plt.show()
                     
sns.regplot(x="Inflight entertainment", y="satisfaction", 
            data=df_subset, logistic=True, ci=None)
                     
Traceback (most recent call last):
  File "<pyshell#44>", line 1, in <module>
    sns.regplot(x="Inflight entertainment", y="satisfaction",
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 775, in regplot
    plotter.plot(ax, scatter_kws, line_kws)
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 384, in plot
    self.lineplot(ax, line_kws)
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 429, in lineplot
    grid, yhat, err_bands = self.fit_regression(ax)
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 218, in fit_regression
    yhat, yhat_boots = self.fit_statsmodels(grid, GLM,
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 295, in fit_statsmodels
    yhat = reg_func(X, y)
  File "D:\py\python\Lib\site-packages\seaborn\regression.py", line 289, in reg_func
    yhat = model(_y, _x, **kwargs).fit().predict(grid)
  File "D:\py\python\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py", line 326, in __init__
    super().__init__(endog, exog, missing=missing,
  File "D:\py\python\Lib\site-packages\statsmodels\base\model.py", line 270, in __init__
    super().__init__(endog, exog, **kwargs)
  File "D:\py\python\Lib\site-packages\statsmodels\base\model.py", line 95, in __init__
    self.data = self._handle_data(endog, exog, missing, hasconst,
  File "D:\py\python\Lib\site-packages\statsmodels\base\model.py", line 135, in _handle_data
    data = handle_data(endog, exog, missing, hasconst, **kwargs)
  File "D:\py\python\Lib\site-packages\statsmodels\base\data.py", line 675, in handle_data
    return klass(endog, exog=exog, missing=missing, hasconst=hasconst,
  File "D:\py\python\Lib\site-packages\statsmodels\base\data.py", line 84, in __init__
    self.endog, self.exog = self._convert_endog_exog(endog, exog)
  File "D:\py\python\Lib\site-packages\statsmodels\base\data.py", line 509, in _convert_endog_exog
    raise ValueError("Pandas data cast to numpy dtype of object. "
ValueError: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).
y_pred = clf.predict(X_test)
                     
y_pred
                     
array(['satisfied', 'dissatisfied', 'dissatisfied', ..., 'dissatisfied',
       'dissatisfied', 'dissatisfied'], shape=(38847,), dtype=object)
clf.predict_proba(X_test)
                     
array([[0.14257646, 0.85742354],
       [0.55008251, 0.44991749],
       [0.89989529, 0.10010471],
       ...,
       [0.89989529, 0.10010471],
       [0.76826369, 0.23173631],
       [0.55008251, 0.44991749]], shape=(38847, 2))
clf.predict(X_test)
                     
array(['satisfied', 'dissatisfied', 'dissatisfied', ..., 'dissatisfied',
       'dissatisfied', 'dissatisfied'], shape=(38847,), dtype=object)
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
                     
Traceback (most recent call last):
  File "<pyshell#49>", line 1, in <module>
    print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
NameError: name 'metrics' is not defined
from sklearn.metrics import accuracy_score
                     
# 如果您使用 from sklearn import metrics
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))

# 或者如果您使用 from sklearn.metrics import accuracy_score
print("Accuracy:", "%.6f" % accuracy_score(y_test, y_pred))
                     
SyntaxError: multiple statements found while compiling a single statement
from sklearn.metrics import accuracy_score
                     
print("Accuracy:", "%.6f" % accuracy_score(y_test, y_pred))
                     
Accuracy: 0.801529
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
                     
Traceback (most recent call last):
  File "<pyshell#54>", line 1, in <module>
    print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
NameError: name 'metrics' is not defined
from sklearn import metrics
                     
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
                     
Traceback (most recent call last):
  File "<pyshell#56>", line 1, in <module>
    print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
  File "D:\py\python\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 2247, in precision_score
    p, _, _, _ = precision_recall_fscore_support(
  File "D:\py\python\Lib\site-packages\sklearn\utils\_param_validation.py", line 189, in wrapper
    return func(*args, **kwargs)
  File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 1830, in precision_recall_fscore_support
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
  File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 1604, in _check_set_wise_labels
    raise ValueError(
ValueError: pos_label=1 is not a valid label. It should be one of ['dissatisfied', 'satisfied']
from sklearn import metricsprint("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))

Traceback (most recent call last):
  File "<pyshell#56>", line 1, in <module>
    print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
  File "D:\py\python\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 2247, in precision_score
    p, _, _, _ = precision_recall_fscore_support(
  File "D:\py\python\Lib\site-packages\sklearn\utils\_param_validation.py", line 189, in wrapper
    return func(*args, **kwargs)
  File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 1830, in precision_recall_fscore_support
...     labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
...   File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 1604, in _check_set_wise_labels
...     raise ValueError(
... ValueError: pos_label=1 is not a valid label. It should be one of ['dissatisfied', 'satisfied']
SyntaxError: invalid syntax
>>> from sklearn import metrics
>>> print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred, pos_label='satisfied'))
Precision: 0.816142
>>> print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
Traceback (most recent call last):
  File "<pyshell#60>", line 1, in <module>
    print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
  File "D:\py\python\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 2429, in recall_score
    _, r, _, _ = precision_recall_fscore_support(
  File "D:\py\python\Lib\site-packages\sklearn\utils\_param_validation.py", line 189, in wrapper
    return func(*args, **kwargs)
  File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 1830, in precision_recall_fscore_support
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
  File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 1604, in _check_set_wise_labels
    raise ValueError(
ValueError: pos_label=1 is not a valid label. It should be one of ['dissatisfied', 'satisfied']
>>> print(metrics.classification_report(y_test, y_pred))
... 
              precision    recall  f1-score   support

dissatisfied       0.78      0.78      0.78     17639
   satisfied       0.82      0.82      0.82     21208

    accuracy                           0.80     38847
   macro avg       0.80      0.80      0.80     38847
weighted avg       0.80      0.80      0.80     38847

>>> print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred, pos_label='satisfied'))
Precision: 0.816142
>>> print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred, pos_label='satisfied'))
Recall: 0.821530
>>> print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred, pos_label='satisfied'))
F1 Score: 0.818827
>>> cm = metrics.confusion_matrix(y_test, y_pred, labels=clf.classes_)
>>> disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
>>> disp.plot()
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at 0x0000021A3E8E0A50>
>>> dfg = df_subset.groupby("satisfaction").agg({"Departure Delay in Minutes": "mean"})
>>> dfg.plot(kind='bar', title='Mean Departure Delay in Minutes', ylabel='Mean Departure Delay in Minutes',xlabel='satistifaction', figsize=(6, 5))
<Axes: title={'center': 'Mean Departure Delay in Minutes'}, xlabel='satistifaction', ylabel='Mean Departure Delay in Minutes'>
