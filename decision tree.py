Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
file = 'Churn_Modelling.csv'
df_original = pd.read_csv(file)
Traceback (most recent call last):
  File "<pyshell#9>", line 1, in <module>
    df_original = pd.read_csv(file)
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
FileNotFoundError: [Errno 2] No such file or directory: 'Churn_Modelling.csv'
file = r'C:\Users\Administrator\Desktop\Churn_Modelling.csv'file = 'Churn_Modelling.csv'
df_original = pd.read_csv(file)
Traceback (most recent call last):
  File "<pyshell#9>", line 1, in <module>
    df_original = pd.read_csv(file)
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
FileNotFoundError: [Errno 2] No such file or directory: 'Churn_Modelling.csv'
SyntaxError: invalid syntax
df_original = pd.read_csv(file)
Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    df_original = pd.read_csv(file)
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
FileNotFoundError: [Errno 2] No such file or directory: 'Churn_Modelling.csv'
file = r'C:\Users\Administrator\Desktop\Churn_Modelling.csv'
df_original = pd.read_csv(file)
df_original.head()
   RowNumber  CustomerId   Surname  ...  IsActiveMember EstimatedSalary Exited
0          1    15634602  Hargrave  ...               1       101348.88      1
1          2    15647311      Hill  ...               1       112542.58      0
2          3    15619304      Onio  ...               0       113931.57      1
3          4    15701354      Boni  ...               0        93826.63      0
4          5    15737888  Mitchell  ...               1        79084.10      0

[5 rows x 14 columns]
df_original['Exited'].value_counts()
Exited
0    7963
1    2037
Name: count, dtype: int64
avg_churned_bal = df_original[df_original['Exited']==1]['Balance'].mean()
avg_churned_bal
np.float64(91108.53933726068)
churn_df = df_original.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender'],
                            axis=1)
churn_df.head()
   CreditScore Geography  Age  ...  IsActiveMember  EstimatedSalary  Exited
0          619    France   42  ...               1        101348.88       1
1          608     Spain   41  ...               1        112542.58       0
2          502    France   42  ...               0        113931.57       1
3          699    France   39  ...               0         93826.63       0
4          850     Spain   43  ...               1         79084.10       0

[5 rows x 10 columns]
churn_df = pd.get_dummies(churn_df, drop_first=True)
churn_df.head()
   CreditScore  Age  Tenure  ...  Exited  Geography_Germany  Geography_Spain
0          619   42       2  ...       1              False            False
1          608   41       1  ...       0              False             True
2          502   42       8  ...       1              False            False
3          699   39       1  ...       0              False            False
4          850   43       2  ...       0              False             True

[5 rows x 11 columns]
y = churn_df['Exited']
X = churn_df.copy()
X = X.drop('Exited', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25, stratify=y,
                                                    random_state=42)
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)
DecisionTreeClassifier(random_state=0)
dt_pred = decision_tree.predict(X_test)
print("Accuracy:", "%.3f" % accuracy_score(y_test, dt_pred))
Accuracy: 0.790
print("Precision:", "%.3f" % precision_score(y_test, dt_pred))
Precision: 0.486
print("Recall:", "%.3f" % recall_score(y_test, dt_pred))
Recall: 0.503
print("F1 Score:", "%.3f" % f1_score(y_test, dt_pred))
F1 Score: 0.494
>>> def conf_matrix_plot(model, x_data, y_data):
...     ```
...     
SyntaxError: invalid syntax
>>> def conf_matrix_plot(model, x_data, y_data):
...     '''
... Accepts as argument model object, X data (test or validate), and y data (test or validate).
... Returns a plot of confusion matrix for predictions on y data.
... '''
...     model_pred = model.predict(x_data)
...     cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
...     disp = ConfusionMatrixDisplay(confusion_matrix=cm,
...                                   display_labels=model.classes_)
...     disp.plot(values_format='')  # `values_format=''` suppresses scientific notation
...     plt.show()
...     conf_matrix_plot(decision_tree, X_test, y_test)
... plt.figure(figsize=(15,12))
SyntaxError: invalid syntax
>>> plt.figure(figsize=(15,12))
<Figure size 1500x1200 with 0 Axes>
>>> plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns,
...           class_names={0:'stayed', 1:'churned'}, filled=True);
[Text(0.5, 0.875, 'Age <= 42.5\ngini = 0.324\nsamples = 7500\nvalue = [5972, 1528]\nclass = stayed'), Text(0.25, 0.625, 'NumOfProducts <= 2.5\ngini = 0.211\nsamples = 5350\nvalue = [4708, 642]\nclass = stayed'), Text(0.375, 0.75, 'True  '), Text(0.125, 0.375, 'NumOfProducts <= 1.5\ngini = 0.188\nsamples = 5223\nvalue = [4674, 549]\nclass = stayed'), Text(0.0625, 0.125, '\n  (...)  \n'), Text(0.1875, 0.125, '\n  (...)  \n'), Text(0.375, 0.375, 'Balance <= 55948.91\ngini = 0.392\nsamples = 127\nvalue = [34, 93]\nclass = churned'), Text(0.3125, 0.125, '\n  (...)  \n'), Text(0.4375, 0.125, '\n  (...)  \n'), Text(0.75, 0.625, 'IsActiveMember <= 0.5\ngini = 0.485\nsamples = 2150\nvalue = [1264, 886]\nclass = stayed'), Text(0.625, 0.75, '  False'), Text(0.625, 0.375, 'Age <= 50.5\ngini = 0.484\nsamples = 964\nvalue = [396, 568]\nclass = churned'), Text(0.5625, 0.125, '\n  (...)  \n'), Text(0.6875, 0.125, '\n  (...)  \n'), Text(0.875, 0.375, 'NumOfProducts <= 2.5\ngini = 0.392\nsamples = 1186\nvalue = [868, 318]\nclass = stayed'), Text(0.8125, 0.125, '\n  (...)  \n'), Text(0.9375, 0.125, '\n  (...)  \n')]
>>> plt.show()
