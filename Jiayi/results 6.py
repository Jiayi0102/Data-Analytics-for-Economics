Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import seaborn as sns
penguins = pd.read_csv("C:\Users\Administrator\Desktop\penguins.csv")
SyntaxError: incomplete input
penguins = pd.read_csv("C:\\Users\\Administrator\\Desktop\\penguins.csv")penguins = pd.read_csv("C:\Users\Administrator\Desktop\penguins.csv")
SyntaxError: incomplete input
SyntaxError: invalid syntax
penguins = pd.read_csv("C:\\Users\\Administrator\\Desktop\\penguins.csv")
penguins = pd.read_csv(r"C:\Users\Administrator\Desktop\penguins.csv")
penguins = pd.read_csv("C:/Users/Administrator/Desktop/penguins.csv")
penguins.head()
  species     island  bill_length_mm  ...  flipper_length_mm  body_mass_g     sex
0  Adelie  Torgersen            39.1  ...              181.0       3750.0    male
1  Adelie  Torgersen            39.5  ...              186.0       3800.0  female
2  Adelie  Torgersen            40.3  ...              195.0       3250.0  female
3  Adelie  Torgersen             NaN  ...                NaN          NaN     NaN
4  Adelie  Torgersen            36.7  ...              193.0       3450.0  female

[5 rows x 7 columns]
penguins = penguins[["body_mass_g", "bill_length_mm", "sex", "species"]]
penguins.columns = ["body_mass_g", "bill_length_mm", "gender", "species"]
penguins.dropna(inplace=True)
penguins.reset_index(inplace=True, drop=True)
penguins.head()
   body_mass_g  bill_length_mm  gender species
0       3750.0            39.1    male  Adelie
1       3800.0            39.5  female  Adelie
2       3250.0            40.3  female  Adelie
3       3450.0            36.7  female  Adelie
4       3650.0            39.3    male  Adelie
penguins_X = penguins[["bill_length_mm", "gender", "species"]]
penguins_y = penguins[["body_mass_g"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(penguins_X, penguins_y,test_size = 0.3, random_state = 42)
ols_formula = "body_mass_g ~ bill_length_mm + C(gender) + C(species)"
from statsmodels.formula.api import ols
ols_data = pd.concat([X_train, y_train], axis = 1)
OLS = ols(formula = ols_formula, data = ols_data)
model = OLS.fit()
model.summary()
<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            body_mass_g   R-squared:                       0.850
Model:                            OLS   Adj. R-squared:                  0.847
Method:                 Least Squares   F-statistic:                     322.6
Date:                Tue, 18 Mar 2025   Prob (F-statistic):           1.31e-92
Time:                        22:55:12   Log-Likelihood:                -1671.7
No. Observations:                 233   AIC:                             3353.
Df Residuals:                     228   BIC:                             3371.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
===========================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------
Intercept                2032.2111    354.087      5.739      0.000    1334.510    2729.913
C(gender)[T.male]         528.9508     55.105      9.599      0.000     420.371     637.531
C(species)[T.Chinstrap]  -285.3865    106.339     -2.684      0.008    -494.920     -75.853
C(species)[T.Gentoo]     1081.6246     94.953     11.391      0.000     894.526    1268.723
bill_length_mm             35.5505      9.493      3.745      0.000      16.845      54.256
==============================================================================
Omnibus:                        0.339   Durbin-Watson:                   1.948
Prob(Omnibus):                  0.844   Jarque-Bera (JB):                0.436
Skew:                           0.084   Prob(JB):                        0.804
Kurtosis:                       2.871   Cond. No.                         798.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
>>> _pred = model.predict(X_test)
... 
>>> y_pred = model.predict(X_test)
>>> test_mse = mean_squared_error(y_test, y_test_pred)
Traceback (most recent call last):
  File "<pyshell#25>", line 1, in <module>
    test_mse = mean_squared_error(y_test, y_test_pred)
NameError: name 'mean_squared_error' is not defined
