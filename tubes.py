from dis import dis
import pandas as pd

xls = pd.ExcelFile('traintest.xlsx')
df = pd.read_excel(xls, sheet_name='train').drop(['id'], axis=1)
df2 = pd.read_excel(xls, sheet_name='test').drop(['y'], axis=1)

# print(df.describe(include="all"))

# print(df.head())

# Metode Modelling

def euclidean_distance(x, y):
  result = []
  for i in range(len(x)):
    res = ((x['x1'][i] - y['x1'])**2) + ((x['x2'][i] - y['x2'])**2) + ((x['x3'][i] - y['x3'])**2)
    result.append([res**0.5, x['y'][i]])
  return result


def manhattan_distance(x, y):
  result = []
  for i in range(len(x)):
    res = (abs(x['x1'][i] - y['x1'])) + abs((x['x2'][i] - y['x2'])) + abs((x['x3'][i] - y['x3']))
    result.append([res, x['y'][i]])
  return result

# print(df.columns.values[1:])

def normalize(df, column, factor=1):
    result = df.copy()
    for col in column:
        max_value = df[col].max()
        min_value = df[col].min()
        result[col] = ((df[col] - min_value) / (max_value - min_value))*factor
    return result

df = normalize(df, df.columns.values[0:])
dfTest = normalize(df2, df2.columns.values[1:])

def k_euclidean(df, dfTest, k):
  result = []
  for i in range(len(dfTest)):
    distance = euclidean_distance(df, dfTest.iloc[i])
    distance.sort()
    resDist = distance[:k]

    a, b = assign_knn(resDist, k)
    if (a > b):
      result.append([dfTest.iloc[i]['id'], 1])
    else:
      result.append([dfTest.iloc[i]['id'], 0])
  return result

def assign_knn(res, k):
  a,b = 0, 0
  for i in range(k):
    if(res[i][1] == 1):
      a += 1
    else:
      b += 1
  return a,b
  
def k_manhattan(df, dfTest, k):
  result = []
  for i in range(len(dfTest)):
    distance = manhattan_distance(df, dfTest.iloc[i])
    distance.sort()
    resDist = distance[:k]
    # print(resDist)
    a, b = assign_knn(resDist, k)
    if (a > b):
      result.append([dfTest.iloc[i]['id'], 1])
    else:
      result.append([dfTest.iloc[i]['id'], 0])
  print(result)

def kkn(df, dfTest, k):
  euclidean = k_euclidean(df, dfTest, k)
  manhattan = k_manhattan(df, dfTest, k)

  return euclidean, manhattan;


a,b = kkn(df, dfTest, 3)
print(a)
print(b)