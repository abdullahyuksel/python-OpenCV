import pandas as pd
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("dataset.csv")


x = dataset.iloc[:,:3].values
y = dataset.iloc[:,3:].values


lr = LinearRegression()
lr.fit(x,y)

pred = lr.predict(x)

pred_list = []

for i in pred:
    b = 0
    ik = 0
    if i[0] > 0.5:
        b =1
    else:
        b=0
    if i[1] > 0.5:
        ik=1
    else:
        ik=0
    pred_list.append([b,ik])
print(pred_list)
print(y)