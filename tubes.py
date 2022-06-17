import pandas as pd

xls = pd.ExcelFile('traintest.xlsx')
df = pd.read_excel(xls, sheet_name='train')
df2 = pd.read_excel(xls, sheet_name='test').drop(['y'], axis=1)

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
    res = (abs(x['x1'][i] - y['x1']) + abs((x['x2'][i] - y['x2'])) + abs((x['x3'][i] - y['x3'])))
    result.append([res, x['y'][i]])
  return result

# normalisasi dengan metode min-max
def normalize(df, column, factor=1):
    result = df.copy()
    for col in column:
        max_value = df[col].max()
        min_value = df[col].min()
        result[col] = ((df[col] - min_value) / (max_value - min_value))*factor
    return result

# mengambil k dari masing-masing modelling
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
  
def k_manhattan(df, dfTest, k):
  result = []
  for i in range(len(dfTest)):
    distance = manhattan_distance(df, dfTest.iloc[i])
    distance.sort()
    resDist = distance[:k]
    a, b = assign_knn(resDist, k)
    if (a > b):
      result.append([dfTest.iloc[i]['id'], 1])
    else:
      result.append([dfTest.iloc[i]['id'], 0])
  return result

# menentukan 1 atau 0
def assign_knn(res, k):
  a,b = 0, 0
  for i in range(k):
    if(res[i][1] == 1):
      a += 1
    else:
      b += 1
  return a,b

# validasi dataset dengan rumus modelling
def validate(fold, k):
  euc = []
  man = []
  for i in range(len(fold)):
    euclidean, manhattan = kkn(fold[i][0], fold[i][1], k)
    euc.append(check_accurracy(euclidean))
    man.append(check_accurracy(manhattan))
  return euc, man

# cek akurasi (masuk dalam validasi)
def check_accurracy(modeling):
  res = 0
  for i in range(len(modeling)):
    if(modeling[i][1] == df['y'][i]):
      res += 1
  return res/len(modeling)


# metode KKN dilakukan
def kkn(df, dfTest, k):
  euclidean = k_euclidean(df, dfTest, k)
  manhattan = k_manhattan(df, dfTest, k)

  return euclidean, manhattan;


# Start
df = normalize(df, df.columns.values[1:])
dfTest = normalize(df2, df2.columns.values[1:])

# fold untuk kebutuhan validasi
fold1 = (df.iloc[0:98],  df.iloc[98:].drop('y', axis=1))
fold2 = (df.iloc[98:196].reset_index(), pd.concat([df.iloc[0:98].reset_index().drop('y', axis=1), df.iloc[98:].reset_index().drop('y', axis=1)]))
fold3 = (df.iloc[196:].reset_index(), df.iloc[0:98].reset_index())

# hasil validasi
euclidean, manhattan = validate([fold1, fold2, fold3], 3)
print(euclidean)
print(manhattan)

euclideanKNN, manhattanKNN = kkn(df, dfTest, 3)
dfEuc = pd.DataFrame(euclideanKNN, columns=['id', 'y'])
dfMan = pd.DataFrame(manhattanKNN, columns=['id', 'y'])

with pd.ExcelWriter('KNN.xlsx') as writer:  
    dfEuc.to_excel(writer, sheet_name='euclidean', index=False)
    dfMan.to_excel(writer, sheet_name='manhattan', index=False)