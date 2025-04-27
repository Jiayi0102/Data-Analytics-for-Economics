Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
SyntaxError: multiple statements found while compiling a single statement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
churn_df = pd.read_csv("C:\Users\Administrator\Desktop\churn_df.csv")
SyntaxError: incomplete input
churn_df = pd.read_csv(r"C:\Users\Administrator\Desktop\churn_df.csv")
churn_df.head()
   CreditScore  Age  Tenure  ...   Loyalty  Geography_Germany  Geography_Spain
0          619   42       2  ...  0.047619                  0                0
1          608   41       1  ...  0.024390                  0                1
2          502   42       8  ...  0.190476                  0                0
3          699   39       1  ...  0.025641                  0                0
4          850   43       2  ...  0.046512                  0                1

[5 rows x 12 columns]
churn_df['Exited'].value_counts()
Exited
0    7963
1    2037
Name: count, dtype: int64
churn_df = churn_df.drop(['Tenure', 'Age'], axis=1)
churn_df.head()
   CreditScore    Balance  ...  Geography_Germany  Geography_Spain
0          619       0.00  ...                  0                0
1          608   83807.86  ...                  0                1
2          502  159660.80  ...                  0                0
3          699       0.00  ...                  0                0
4          850  125510.82  ...                  0                1

[5 rows x 10 columns]
y = churn_df['Exited']
X = churn_df.copy()
X = X.drop('Exited', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, \
                                                    stratify=y, random_state=42)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
GaussianNB()
y_preds = gnb.predict(X_test)
print('Accuracy:', '%.3f' % accuracy_score(y_test, y_preds))
Accuracy: 0.796
print('Precision:', '%.3f' % precision_score(y_test, y_preds))

Warning (from warnings module):
  File "D:\py\python\Lib\site-packages\sklearn\metrics\_classification.py", line 1565
    _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
Precision: 0.000
print('Recall:', '%.3f' % recall_score(y_test, y_preds))
Recall: 0.000
print('F1 Score:', '%.3f' % f1_score(y_test, y_preds))
F1 Score: 0.000
np.unique(y_preds)
array([0])
X.describe()
        CreditScore        Balance  ...  Geography_Germany  Geography_Spain
count  10000.000000   10000.000000  ...       10000.000000     10000.000000
mean     650.528800   76485.889288  ...           0.250900         0.247700
std       96.653299   62397.405202  ...           0.433553         0.431698
min      350.000000       0.000000  ...           0.000000         0.000000
25%      584.000000       0.000000  ...           0.000000         0.000000
50%      652.000000   97198.540000  ...           0.000000         0.000000
75%      718.000000  127644.240000  ...           1.000000         0.000000
max      850.000000  250898.090000  ...           1.000000         1.000000

[8 rows x 9 columns]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
SyntaxError: multiple statements found while compiling a single statement
scaler = MinMaxScaler()
scaler.fit(X_train)
MinMaxScaler()
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
gnb_scaled = GaussianNB()
gnb_scaled.fit(X_train, y_train)
GaussianNB()
scaled_preds = gnb_scaled.predict(X_test)
print('Accuracy:', '%.3f' % accuracy_score(y_test, scaled_preds))
Accuracy: 0.806
print('Precision:', '%.3f' % precision_score(y_test,scaled_preds))
Precision: 0.544
print('Recall:', '%.3f' % recall_score(y_test, scaled_preds))
Recall: 0.303
print('F1 Score:', '%.3f' % f1_score(y_test, scaled_preds))
F1 Score: 0.389
def conf_matrix_plot(model, x_data, y_data):
    ''
    Accepts as argument model object, X data (test or validate), and y data (test or validate).
    
SyntaxError: invalid syntax
>>> def conf_matrix_plot(model, x_data, y_data):
...     Accepts as argument model object, X data (test or validate), and y data (test or validate).
...     
SyntaxError: invalid syntax
>>> from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
>>> import matplotlib.pyplot as plt
>>> def conf_matrix_plot(model, x_data, y_data):
...     '''
... model_pred = model.predict(x_data)
... cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
... disp = ConfusionMatrixDisplay(confusion_matrix=cm,
... display_labels=model.classes_)
... disp.plot(values_format='')
... plt.show()
... conf_matrix_plot(gnb_scaled, X_test, y_test)
... from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
... import matplotlib.pyplot as plt
... def conf_matrix_plot(model, x_data, y_data):
... '''
...     Accepts as argument model object, X data (test or validate), and y data (test or validate).
...     
SyntaxError: invalid syntax
>>> from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
>>> import matplotlib.pyplot as plt
>>> def conf_matrix_plot(model, x_data, y_data):
...     '''
...     Accepts as argument model object, X data (test or validate), and y data (test or validate).
...     Return a plot of confusion matrix for predictions on y data.
...     '''
...     model_pred = model.predict(x_data)
...     cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
...     disp = ConfusionMatrixDisplay(confusion_matrix=cm,
...                                   display_labels=model.classes_)
...     disp.plot(values_format='')
...     plt.show()
...     conf_matrix_plot(gnb_scaled, X_test, y_test)
