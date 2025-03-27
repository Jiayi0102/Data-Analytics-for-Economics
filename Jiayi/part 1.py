df_original = pd.read_csv(r"C:\Users\Administrator\Desktop\churn_df.csv")
df_original.head()
   CreditScore  Age  Tenure  ...   Loyalty  Geography_Germany  Geography_Spain
0          619   42       2  ...  0.047619                  0                0
1          608   41       1  ...  0.024390                  0                1
2          502   42       8  ...  0.190476                  0                0
3          699   39       1  ...  0.025641                  0                0
4          850   43       2  ...  0.046512                  0                1

[5 rows x 12 columns]
df_original.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   CreditScore        10000 non-null  int64  
 1   Age                10000 non-null  int64  
 2   Tenure             10000 non-null  int64  
 3   Balance            10000 non-null  float64
 4   NumOfProducts      10000 non-null  int64  
 5   HasCrCard          10000 non-null  int64  
 6   IsActiveMember     10000 non-null  int64  
 7   EstimatedSalary    10000 non-null  float64
 8   Exited             10000 non-null  int64  
 9   Loyalty            10000 non-null  float64
 10  Geography_Germany  10000 non-null  int64  
 11  Geography_Spain    10000 non-null  int64  
dtypes: float64(3), int64(9)
memory usage: 937.6 KB
print(df_original.columns)
Index(['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited', 'Loyalty',
       'Geography_Germany', 'Geography_Spain'],
      dtype='object')
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Gender']
existing_columns = [col for col in columns_to_drop if col in df_original.columns]
churn_df = df_original.drop(existing_columns, axis=1)
churn_df = df_original.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender'], axis=1, errors='ignore')
churn_df.head()
   CreditScore  Age  Tenure  ...   Loyalty  Geography_Germany  Geography_Spain
0          619   42       2  ...  0.047619                  0                0
1          608   41       1  ...  0.024390                  0                1
2          502   42       8  ...  0.190476                  0                0
3          699   39       1  ...  0.025641                  0                0
4          850   43       2  ...  0.046512                  0                1

[5 rows x 12 columns]
churn_df['Loyalty'] = churn_df['Tenure'] / churn_df['Age']
churn_df.head()
   CreditScore  Age  Tenure  ...   Loyalty  Geography_Germany  Geography_Spain
0          619   42       2  ...  0.047619                  0                0
1          608   41       1  ...  0.024390                  0                1
2          502   42       8  ...  0.190476                  0                0
3          699   39       1  ...  0.025641                  0                0
4          850   43       2  ...  0.046512                  0                1

[5 rows x 12 columns]
