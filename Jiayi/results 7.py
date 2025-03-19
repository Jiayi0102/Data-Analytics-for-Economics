Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> import statsmodels.api as sm
>>> import pylab as py
>>> np.random.seed(0)
>>> data_normal = np.random.normal(0, 1, 100)
>>> data_exponential = np.random.exponential(scale=1.0, size=100)
>>> data_t = np.random.standard_t(df=3, size=100)
>>> fig, axes = py.subplots(1, 3, figsize=(18, 5))
>>> sm.qqplot(data_normal, line='45', ax=axes[0])
<Figure size 1800x500 with 3 Axes>
>>> axes[0].set_title("Normal Distribution")
Text(0.5, 1.0, 'Normal Distribution')
>>> sm.qqplot(data_exponential, line='45', ax=axes[1])
<Figure size 1800x500 with 3 Axes>
>>> axes[1].set_title("Exponential Distribution")
Text(0.5, 1.0, 'Exponential Distribution')
>>> sm.qqplot(data_t, line='45', ax=axes[2])
<Figure size 1800x500 with 3 Axes>
>>> axes[2].set_title("t-Distribution (df=3)")
Text(0.5, 1.0, 't-Distribution (df=3)')
>>> py.show()
