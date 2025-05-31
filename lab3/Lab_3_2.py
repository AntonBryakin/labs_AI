# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
print(df.head())

x = df.iloc[:, :4].values
y = df.iloc[:, 4].values

tmp = {"Iris-setosa": 0, "Iris-versicolor": 1}
y = np.array([tmp[tmp_arr] for tmp_arr in y])

x_tensor = torch.tensor(x, dtype = torch.float32)
y_tensor = torch.tensor(y, dtype = torch.long)

linear = nn.Linear(4, 2)
loss_fn = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

for i in range(100):
    pred = linear(x_tensor)
    loss = loss_fn(pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if(i + 1) % 5 == 0:
        print(f'{i + 1} / {100} err: {loss.item():4f}')
        
with torch.no_grad():  
    predictions = linear(x_tensor)
    _, predicted_classes = torch.max(predictions, 1)
    
t = torch.tensor(y, dtype=torch.long)  

print("Предсказанные классы:", predicted_classes)
print("Метки:", t)