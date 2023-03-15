import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')

############################ Load the Titanic dataset ############################
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

############################# EDA ############################
print('train:',train_data.columns.values)
# train: ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch''Ticket' 'Fare' 'Cabin' 'Embarked']
# 숫자데이터 : PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
print('test:',test_data.columns.values)
# test: ['PassengerId' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']
# 숫자데이터 : PassengerId      Pclass         Age       SibSp       Parch        Fare

############################## 변수설명 #############################
# PassengerId : 각 승객의 고유 번호
# Survived : 생존 여부(종속 변수) 0 = 사망 1 = 생존
# Pclass : 객실 등급 - 승객의 사회적, 경제적 지위
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# Name : 이름
# Sex : 성별
# Age : 나이
# SibSp : 동반한 Sibling(형제자매)와 Spouse(배우자)의 수
# Parch : 동반한 Parent(부모) Child(자식)의 수
# Ticket : 티켓의 고유넘버
# Fare : 티켓의 요금
# Cabin : 객실 번호
# Embarked : 승선한 항 C = Cherbourg Q = Queenstown S = Southampton

############################## head #############################
print(train_data.head())
print(test_data.head())
#    PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
# 0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
# 1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
# 2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
# 3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
# 4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S
#    PassengerId  Pclass                                          Name     Sex   Age  SibSp  Parch   Ticket     Fare Cabin Embarked
# 0          892       3                              Kelly, Mr. James    male  34.5      0      0   330911   7.8292   NaN        Q
# 1          893       3              Wilkes, Mrs. James (Ellen Needs)  female  47.0      1      0   363272   7.0000   NaN        S
# 2          894       2                     Myles, Mr. Thomas Francis    male  62.0      0      0   240276   9.6875   NaN        Q
# 3          895       3                              Wirz, Mr. Albert    male  27.0      0      0   315154   8.6625   NaN        S
# 4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female  22.0      1      1  3101298  12.2875   NaN        S

############################## information #############################
print(train_data.info())
print(test_data.info())
# .info() 메써드로 확인 
# train 데이터는 891행 / test 데이터는 418행
# 그 중 Age와 Cabin만 데이터 개수가 다르다. Age와 Cabin을 제외하면 NaN 값은 더 없다.
# 물론 값이 없는 승객은 그냥 데이터에서 빼버려도 되지만, 이 문제는 데이터가 충분하지 않기 때문에 고민중.

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  891 non-null    int64
#  1   Survived     891 non-null    int64
#  2   Pclass       891 non-null    int64
#  3   Name         891 non-null    object
#  4   Sex          891 non-null    object
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
#  8   Ticket       891 non-null    object
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object
#  11  Embarked     889 non-null    object
# dtypes: float64(2), int64(5), object(5)
# 소수점 숫자가 있는 2개의 열, 정수가 있는 5개의 열, 텍스트 또는 혼합 데이터가 있는 5개의 열

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 418 entries, 0 to 417
# Data columns (total 11 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  418 non-null    int64
#  1   Pclass       418 non-null    int64
#  2   Name         418 non-null    object
#  3   Sex          418 non-null    object
#  4   Age          332 non-null    float64
#  5   SibSp        418 non-null    int64
#  6   Parch        418 non-null    int64
#  7   Ticket       418 non-null    object
#  8   Fare         417 non-null    float64
#  9   Cabin        91 non-null     object
#  10  Embarked     418 non-null    object
# dtypes: float64(2), int64(4), object(5)
# 소수점 숫자가 있는 2개의 열, 정수가 있는 4개의 열, 텍스트 또는 혼합 데이터가 있는 5개의 열

############################## describe #############################
print(train_data.describe())
print(test_data.describe())
#        PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
# count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
# mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
# std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
# min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
# 25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
# 50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
# 75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
# max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
#        PassengerId      Pclass         Age       SibSp       Parch        Fare
# count   418.000000  418.000000  332.000000  418.000000  418.000000  417.000000
# mean   1100.500000    2.265550   30.272590    0.447368    0.392344   35.627188
# std     120.810458    0.841838   14.181209    0.896760    0.981429   55.907576
# min     892.000000    1.000000    0.170000    0.000000    0.000000    0.000000
# 25%     996.250000    1.000000   21.000000    0.000000    0.000000    7.895800
# 50%    1100.500000    3.000000   27.000000    0.000000    0.000000   14.454200
# 75%    1204.750000    3.000000   39.000000    1.000000    0.000000   31.500000
# max    1309.000000    3.000000   76.00000e0    8.000000    9.000000  512.329200

############################## Sex,Embarked,Sibsp,Parch survived ratio  #############################
print(train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_data[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#       Sex  Survived
# 0  female  0.742038
# 1    male  0.188908

# Embarked  Survived
# 0        C  0.553571
# 1        Q  0.389610
# 2        S  0.336957

#    SibSp  Survived
# 1      1  0.535885
# 2      2  0.464286
# 0      0  0.345395
# 3      3  0.250000
# 4      4  0.166667
# 5      5  0.000000
# 6      8  0.000000

#    Parch  Survived
# 3      3  0.600000
# 1      1  0.550847
# 2      2  0.500000
# 0      0  0.343658
# 5      5  0.200000
# 4      4  0.000000
# 6      6  0.000000

print(train_data.isnull().sum()) 
# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             1
# Cabin          687
# Embarked         2
# dtype: int64

############################# 시각화 ############################
plt.style.use('ggplot')
sns.set()
sns.set_palette("Set2")

def chart(dataset, feature):
    survived = dataset[dataset['Survived'] == 1][feature].value_counts()
    dead = dataset[dataset['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True)
# plt.show()

########################### Parch ###################################
chart(train_data, 'Parch')
# 애매하다 차이가 있긴한데 영향이 있는지는 잘 모르겠다.

############################ Pclass ##################################
chart(train_data, 'Pclass')
# 1등석에 있던 승객들은 오히려 사망한 사람보다 생존한 사람이 많고, 
# 2등석은 대략 생존율이 50%정도 되는 것 같다. 
# 반면 3등석은 생존한 사람보다 사망한 사람이 훨씬 많다. 
# 이걸 보아 Pclass라는 특성은 승객의 생사를 예측하는 데에 
# 영향을 끼친다는 것을 확인할 수 있다. 

############################ Sex ##################################
chart(train_data, 'Sex')
# 성별에 따른 분류.
# 여성은 생존률이 매우 높은 반면, 남성은 사망한 사람이 훨씬 많다. 
# 사고 당시에 남성보다는 여성을 우선적으로 살린 것으로 보인다.

############################# SibSp #################################
chart(train_data, 'SibSp')
# temp = train_data[(train_data['SibSp'] > 2)]
# chart(temp, 'SibSp')
# - chart(train, 'SibSp')
# 자매와 배우자의 수에 따라 도시한 그래프. 
# SibSp 값이 0인 사람보다는, 비율상 1이나 2인 사람들이 더욱 많이 생존.
# 하지만, 3 이상부터는 잘 보이지 않는다. 
temp = train_data[(train_data['SibSp'] > 2)]
chart(temp, 'SibSp')
# SibSp 값이 3 이상인 사람들은 생존율이 높지 않다는 것을 확인. (영향이 있어보인다)

############################ Embarked ##################################
chart(train_data, 'Embarked')
# 승객별 탑승지에 따른 생존율. 
# 탑승지의 차이라기에는 생각보다 차이가 많이 난다. 
# 지역별로 부유한 도시와 가난한 도시가 있을수도 있을 것 같다고 생각. 
# 탑승지별로 1등석, 2등석, 3등석의 수를 한번 확인.
S = train_data[train_data['Embarked'] == 'S']['Pclass'].value_counts()
C = train_data[train_data['Embarked'] == 'C']['Pclass'].value_counts()
Q = train_data[train_data['Embarked'] == 'Q']['Pclass'].value_counts()
df = pd.DataFrame([S, C, Q])
df.index = ['S', 'C', 'Q']
df.plot(kind='bar', stacked=True)
# 1등석의 비율이 탑승지별로 다른 것을 확인. 
# Embarked가 C인 사람들은 1등석 비율이 거의 절반. 
# 이전 그래프에서 탑승지가 C였던 사람들의 생존률이 
# 거의 50퍼센트에 가깝게 나왔다. 영향이 있을 것 같다.
# plt.show()

############################ 2.데이터 전처리  ##################################

# fare가 왜 결측지가 있는지 모르겠지만, 1개라 평균값으로 채우기.
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# 생존 예측에 큰 영향을 미치지 않을 것으로 예상되는 'PassengerId', 'Name' 및 'Ticket' 열을 삭제
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# 'male'을 0으로, 'female'을 1로 바꾸어 'Sex' 열을 숫자 열로 변환
train_data['Sex'] = train_data['Sex'].replace({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].replace({'male': 0, 'female': 1})

# 'Age' 열의 누락된 값을 데이터 세트의 평균 연령으로 채우기
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

# 'SibSp' 및 'Parch' 열을 추가하여 새 열 'FamilySize'를 생성.
# 승객이 혼자 여행하는지 또는 가족과 함께 여행하는지를 나타내는 새 열 'IsAlone'을 생성.
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
train_data['IsAlone'] = 0
test_data['IsAlone'] = 0
train_data.loc[train_data['FamilySize'] == 1, 'IsAlone'] = 1
test_data.loc[test_data['FamilySize'] == 1, 'IsAlone'] = 1

# 'Embarked' 열의 누락된 값을 가장 일반적인 값('S')으로 채우기.
train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')

# 'S'를 0으로, 'C'를 1로, 'Q'를 2로 바꾸어 'Embarked' 열을 숫자 열로 변환.
train_data['Embarked'] = train_data['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
test_data['Embarked'] = test_data['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})

# 승객에게 객실이 있는지 여부를 나타내는 새 열 'CabinBool'을 생성.
train_data['CabinBool'] = (train_data['Cabin'].notnull().astype('int'))
test_data['CabinBool'] = (test_data['Cabin'].notnull().astype('int'))

# 누락된 값이 많고 생존 예측에 큰 영향을 미치지 않을 것으로 예상되므로 'Cabin' 열을 삭제
train_data = train_data.drop(['Cabin'], axis=1)
test_data = test_data.drop(['Cabin'], axis=1)