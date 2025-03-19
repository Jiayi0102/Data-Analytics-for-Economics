Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
data = pd.read_csv("C:\Users\Administrator\Desktop\tiktok_dataset.csv")
SyntaxError: incomplete input
data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\tiktok_dataset.csv")data = pd.read_csv("C:\Users\Administrator\Desktop\tiktok_dataset.csv")
SyntaxError: incomplete input
SyntaxError: invalid syntax
data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\tiktok_dataset.csv")
data.head(10)
    # claim_status  ...  video_download_count  video_comment_count
0   1        claim  ...                   1.0                  0.0
1   2        claim  ...                1161.0                684.0
2   3        claim  ...                 833.0                329.0
3   4        claim  ...                1234.0                584.0
4   5        claim  ...                 547.0                152.0
5   6        claim  ...                4293.0               1857.0
6   7        claim  ...                8616.0               5446.0
7   8        claim  ...                  22.0                 11.0
8   9        claim  ...                  53.0                 27.0
9  10        claim  ...                4104.0               2540.0

[10 rows x 12 columns]
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19382 entries, 0 to 19381
Data columns (total 12 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   #                         19382 non-null  int64  
 1   claim_status              19084 non-null  object 
 2   video_id                  19382 non-null  int64  
 3   video_duration_sec        19382 non-null  int64  
 4   video_transcription_text  19084 non-null  object 
 5   verified_status           19382 non-null  object 
 6   author_ban_status         19382 non-null  object 
 7   video_view_count          19084 non-null  float64
 8   video_like_count          19084 non-null  float64
 9   video_share_count         19084 non-null  float64
 10  video_download_count      19084 non-null  float64
 11  video_comment_count       19084 non-null  float64
dtypes: float64(5), int64(3), object(4)
memory usage: 1.8+ MB
data.describe()
                  #      video_id  ...  video_download_count  video_comment_count
count  19382.000000  1.938200e+04  ...          19084.000000         19084.000000
mean    9691.500000  5.627454e+09  ...           1049.429627           349.312146
std     5595.245794  2.536440e+09  ...           2004.299894           799.638865
min        1.000000  1.234959e+09  ...              0.000000             0.000000
25%     4846.250000  3.430417e+09  ...              7.000000             1.000000
50%     9691.500000  5.618664e+09  ...             46.000000             9.000000
75%    14536.750000  7.843960e+09  ...           1156.250000           292.000000
max    19382.000000  9.999873e+09  ...          14994.000000          9599.000000

[8 rows x 8 columns]
# Check unique values and their counts for claim_status
data['claim_status'].value_counts()
claim_status
claim      9608
opinion    9476
Name: count, dtype: int64
data.head()
   # claim_status  ...  video_download_count  video_comment_count
0  1        claim  ...                   1.0                  0.0
1  2        claim  ...                1161.0                684.0
2  3        claim  ...                 833.0                329.0
3  4        claim  ...                1234.0                584.0
4  5        claim  ...                 547.0                152.0

[5 rows x 12 columns]
claim_videos = data[data['claim_status'] == 'claim']
avg_view_count = claim_videos['video_view_count'].mean()
print(f"Average view count for 'claim' videos: {avg_view_count}")
Average view count for 'claim' videos: 501029.4527477102
opinion_videos = df[df['claim_status'] == 'opinion']
Traceback (most recent call last):
  File "<pyshell#23>", line 1, in <module>
    opinion_videos = df[df['claim_status'] == 'opinion']
NameError: name 'df' is not defined
opinion_videos = df[df['claim_status'] == 'opinion']
Traceback (most recent call last):
  File "<pyshell#24>", line 1, in <module>
    opinion_videos = df[df['claim_status'] == 'opinion']
NameError: name 'df' is not defined
import pandas as pd
df = pd.read_csv(r"C:\Users\Administrator\Desktop\tiktok_dataset.csv")
opinion_videos = df[df['claim_status'] == 'opinion']
avg_view_count = opinion_videos['video_view_count'].mean()
print(f"Average view count of videos with 'opinion' status: {avg_view_count:.2f}")
Average view count of videos with 'opinion' status: 4956.43
total_opinion_videos = len(opinion_videos)
print(f"Total number of videos with 'opinion' status: {total_opinion_videos}")
Total number of videos with 'opinion' status: 9476
print("\nDistribution statistics of view counts for opinion videos:")

Distribution statistics of view counts for opinion videos:
print(opinion_videos['video_view_count'].describe())
count    9476.000000
mean     4956.432250
std      2885.907219
min        20.000000
25%      2467.000000
50%      4953.000000
75%      7447.250000
max      9998.000000
Name: video_view_count, dtype: float64
group_counts = df.groupby(['claim_status', 'author_ban_status']).size().reset_index(name='count')
group_counts = group_counts.sort_values('count', ascending=False)
print("Counts for each combination of claim status and author ban status:")
Counts for each combination of claim status and author ban status:
print(group_counts)
  claim_status author_ban_status  count
3      opinion            active   8817
0        claim            active   6566
2        claim      under review   1603
1        claim            banned   1439
5      opinion      under review    463
4      opinion            banned    196
total_videos = len(df)
group_counts['percentage'] = (group_counts['count'] / total_videos * 100).round(2)
print("\nPercentages for each combination:")

Percentages for each combination:
print(group_counts[['claim_status', 'author_ban_status', 'count', 'percentage']])
  claim_status author_ban_status  count  percentage
3      opinion            active   8817       45.49
0        claim            active   6566       33.88
2        claim      under review   1603        8.27
1        claim            banned   1439        7.42
5      opinion      under review    463        2.39
4      opinion            banned    196        1.01
print("\nCrosstab view:")

Crosstab view:
crosstab = pd.crosstab(df['claim_status'], df['author_ban_status'], margins=True, margins_name='Total')
print(crosstab)
author_ban_status  active  banned  under review  Total
claim_status                                          
claim                6566    1439          1603   9608
opinion              8817     196           463   9476
Total               15383    1635          2066  19084
median_shares = df.groupby('author_ban_status')['video_share_count'].median().reset_index()
median_shares.columns = ['Author Ban Status', 'Median Share Count']
median_shares = median_shares.sort_values('Median Share Count', ascending=False)
print("Median video share count by author ban status:")
Median video share count by author ban status:
print(median_shares)
  Author Ban Status  Median Share Count
1            banned             14468.0
2      under review              9444.0
0            active               437.0
status_counts = df['author_ban_status'].value_counts().reset_index()
status_counts.columns = ['Author Ban Status', 'Number of Videos']
combined = pd.merge(median_shares, status_counts, on='Author Ban Status')

print("\nDetailed statistics for each author ban status:")

Detailed statistics for each author ban status:
for status in df['author_ban_status'].unique():
    status_data = df[df['author_ban_status'] == status]['video_share_count']
    print(f"\nAuthor Ban Status: {status}")
    print(f"Number of videos: {len(status_data)}")
    print(f"Median share count: {status_data.median()}")
    print(f"Mean share count: {status_data.mean():.2f}")
    print(f"25th percentile: {status_data.quantile(0.25)}")
    print(f"75th percentile: {status_data.quantile(0.75)}")
    print(f"Min: {status_data.min()}, Max: {status_data.max()}")
stats_by_status = df.groupby('author_ban_status').agg({
    
SyntaxError: invalid syntax
stats_by_status = df.groupby('author_ban_status').agg({'video_view_count': ['count', 'mean', 'median'],'video_like_count': ['count', 'mean', 'median'],'video_share_count': ['count', 'mean', 'median']})
stats_by_status.columns = ['_'.join(col).strip() for col in stats_by_status.columns.values]
stats_by_status = stats_by_status.reset_index()
for col in stats_by_status.columns:if 'mean' in col:stats_by_status[col] = stats_by_status[col].round(2)
SyntaxError: multiple statements found while compiling a single statement
for col in stats_by_status.columns:
if 'mean' in col:
SyntaxError: expected an indented block after 'for' statement on line 1
for col in stats_by_status.columns:
if 'mean' in col:stats_by_status[col] = stats_by_status[col].round(2)
print("Statistics by Author Ban Status:")
SyntaxError: expected an indented block after 'for' statement on line 1
print(stats_by_status)
                   video_view_count_count  ...  video_share_count_median
author_ban_status                          ...                          
active                              15383  ...                     437.0
banned                               1635  ...                   14468.0
under review                         2066  ...                    9444.0

[3 rows x 9 columns]
df['likes_per_view'] = df['video_like_count'] / df['video_view_count']
df['comments_per_view'] = df['video_comment_count'] / df['video_view_count']
df['shares_per_view'] = df['video_share_count'] / df['video_view_count']
df.replace([np.inf, -np.inf], np.nan, inplace=True)
SyntaxError: multiple statements found while compiling a single statement
print(df[['video_view_count', 'video_like_count', 'video_comment_count', 'video_share_count', 
         'likes_per_view', 'comments_per_view', 'shares_per_view']].head())
Traceback (most recent call last):
  File "<pyshell#73>", line 1, in <module>
    print(df[['video_view_count', 'video_like_count', 'video_comment_count', 'video_share_count',
  File "D:\py\python\Lib\site-packages\pandas\core\frame.py", line 4108, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
  File "D:\py\python\Lib\site-packages\pandas\core\indexes\base.py", line 6200, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "D:\py\python\Lib\site-packages\pandas\core\indexes\base.py", line 6252, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: "['likes_per_view', 'comments_per_view', 'shares_per_view'] not in index"
print(df[['video_view_count', 'video_like_count', 'video_comment_count', 'video_share_count','likes_per_view', 'comments_per_view', 'shares_per_view']].head())
Traceback (most recent call last):
  File "<pyshell#74>", line 1, in <module>
    print(df[['video_view_count', 'video_like_count', 'video_comment_count', 'video_share_count','likes_per_view', 'comments_per_view', 'shares_per_view']].head())
  File "D:\py\python\Lib\site-packages\pandas\core\frame.py", line 4108, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
  File "D:\py\python\Lib\site-packages\pandas\core\indexes\base.py", line 6200, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "D:\py\python\Lib\site-packages\pandas\core\indexes\base.py", line 6252, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: "['likes_per_view', 'comments_per_view', 'shares_per_view'] not in index"
df = df.replace([np.inf, -np.inf], np.nan)df.replace([np.inf, -np.inf], np.nan, inplace=True)
SyntaxError: multiple statements found while compiling a single statement
print(df[['video_view_count', 'video_like_count', 'video_comment_count', 'video_share_count',
         'likes_per_view', 'comments_per_view', 'shares_per_view']].head())
Traceback (most recent call last):
  File "<pyshell#73>", line 1, in <module>
    print(df[['video_view_count', 'video_like_count', 'video_comment_count', 'video_share_count',
  File "D:\py\python\Lib\site-packages\pandas\core\frame.py", line 4108, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
  File "D:\py\python\Lib\site-packages\pandas\core\indexes\base.py", line 6200, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "D:\py\python\Lib\site-packages\pandas\core\indexes\base.py", line 6252, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: "['likes_per_view', 'comments_per_view', 'shares_per_view'] not in index"
print(df[['video_view_count', 'video_like_count', 'video_comment_count', 'video_share_count','likes_per_view', 'comments_per_view', 'shares_per_view']].head())
Traceback (most recent call last):
  File "<pyshell#74>", line 1, in <module>
    print(df[['video_view_count', 'video_like_count', 'video_comment_count', 'video_share_count','likes_per_view', 'comments_per_view', 'shares_per_view']].head())
  File "D:\py\python\Lib\site-packages\pandas\core\frame.py", line 4108, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
  File "D:\py\python\Lib\site-packages\pandas\core\indexes\base.py", line 6200, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "D:\py\python\Lib\site-packages\pandas\core\indexes\base.py", line 6252, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: "['likes_per_view', 'comments_per_view', 'shares_per_view'] not in index"
SyntaxError: invalid syntax
df['likes_per_view'] = df['video_like_count'] / df['video_view_count']
df['comments_per_view'] = df['video_comment_count'] / df['video_view_count']
df['shares_per_view'] = df['video_share_count'] / df['video_view_count']
df = df.replace([np.inf, -np.inf], np.nan)
print(df.columns.tolist())
['#', 'claim_status', 'video_id', 'video_duration_sec', 'video_transcription_text', 'verified_status', 'author_ban_status', 'video_view_count', 'video_like_count', 'video_share_count', 'video_download_count', 'video_comment_count', 'likes_per_view', 'comments_per_view', 'shares_per_view']
print(df[['video_view_count', 'video_like_count', 'likes_per_view']].head())
   video_view_count  video_like_count  likes_per_view
0          343296.0           19425.0        0.056584
1          140877.0           77355.0        0.549096
2          902185.0           97690.0        0.108282
3          437506.0          239954.0        0.548459
4           56167.0           34987.0        0.622910
print(df[['video_view_count', 'video_comment_count', 'comments_per_view']].head())
   video_view_count  video_comment_count  comments_per_view
0          343296.0                  0.0           0.000000
1          140877.0                684.0           0.004855
2          902185.0                329.0           0.000365
3          437506.0                584.0           0.001335
4           56167.0                152.0           0.002706
print(df[['video_view_count', 'video_share_count', 'shares_per_view']].head())
   video_view_count  video_share_count  shares_per_view
0          343296.0              241.0         0.000702
1          140877.0            19034.0         0.135111
2          902185.0             2858.0         0.003168
3          437506.0            34812.0         0.079569
4           56167.0             4110.0         0.073175
df['likes_per_view'] = df['video_like_count'] / df['video_view_count']
df['comments_per_view'] = df['video_comment_count'] / df['video_view_count']
df['shares_per_view'] = df['video_share_count'] / df['video_view_count']
df = df.replace([np.inf, -np.inf], np.nan)
KeyboardInterrupt
numeric_columns = ['video_duration_sec', 'video_view_count', 'video_like_count',
                   'video_share_count', 'video_download_count', 'video_comment_count','likes_per_view', 'comments_per_view', 'shares_per_view']
desc_stats = df[numeric_columns].describe()
SyntaxError: multiple statements found while compiling a single statement
desc_stats = df[numeric_columns].describe()desc_stats = df[numeric_columns].describe()
SyntaxError: multiple statements found while compiling a single statement
SyntaxError: invalid syntax
>>> numeric_columns = ['video_duration_sec', 'video_view_count', 'video_like_count', 
...                   'video_share_count', 'video_download_count', 'video_comment_count',
...                   'likes_per_view', 'comments_per_view', 'shares_per_view']
>>> desc_stats = df[numeric_columns].describe()
... print(desc_stats)
SyntaxError: multiple statements found while compiling a single statement
>>> variance_values = df[numeric_columns].var()
>>> print("\nVariance values:")

Variance values:
>>> print(variance_values)
video_duration_sec      2.634118e+02
video_view_count        1.042601e+11
video_like_count        1.780104e+10
video_share_count       1.026316e+09
video_download_count    4.017218e+06
video_comment_count     6.394223e+05
likes_per_view          2.993110e-02
comments_per_view       1.757083e-06
shares_per_view         2.560020e-03
dtype: float64
>>> missing_values = df[numeric_columns].isna().sum()
>>> print("\nMissing values count:")

Missing values count:
>>> print(missing_values)
video_duration_sec        0
video_view_count        298
video_like_count        298
video_share_count       298
video_download_count    298
video_comment_count     298
likes_per_view          298
comments_per_view       298
shares_per_view         298
dtype: int64
