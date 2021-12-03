import numpy as np
from numpy.core.numeric import NaN
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 9
hidden_size = 20
num_epochs = 5
batch_size = 4
lr = 0.01

### Class titanic
class TitanicDataset(Dataset):
    def __init__(self, path):
        self.path = path
        titanic = pd.read_csv(path, delimiter=',')
        del titanic['PassengerId']
        del titanic['Name']
        del titanic['Ticket']
        del titanic['Cabin']
        titanic['C'] = (titanic['Embarked'] == 'C')*1
        titanic['Q'] = (titanic['Embarked'] == 'Q')*1
        titanic['S'] = (titanic['Embarked'] == 'S')*1
        del titanic['Embarked']
        titanic['Sex'] = (titanic['Sex'] == 'male')*1
        meadianAge = titanic['Age'].median()
        titanic['Age'] = titanic['Age'].replace(np.nan, meadianAge)
        self.x = torch.from_numpy(titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S']].to_numpy(dtype=np.float32))
        self.y = torch.from_numpy(titanic['Survived'].to_numpy(dtype=np.float32))
        self.n_samples = titanic.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


train = TitanicDataset(path='train.csv')
# a, b = train[0] #  0       3    male  22.0      1      0   7.2500        S  0  0  1
# print(a, b)
dl = DataLoader(dataset=train, batch_size=1, shuffle=True)

total_samples = len(train)
n_it = math.ceil(total_samples/4)

### model ###
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


model = NeuralNet1(input_size,hidden_size)
criterion =  nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training loop

for epoch in range(num_epochs):
    for i,(inputs, labels) in enumerate(dl):
        t = inputs.reshape(-1).to(device)
        l = labels.reshape(-1).to(device)

        # forward
        out = model(t)
        loss = criterion(out, l)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 100 == 0:
            print(f'epcoch: {epoch+1}/{num_epochs}, step: {i+1}/{total_samples}, loss: {loss.item():.4f}')


print('---------'*30)


test = TitanicDataset(path='test.csv')
tl = DataLoader(dataset=test, batch_size=1, shuffle=False)

with torch.no_grad():
    for i, (x,_) in enumerate(tl):
        x = x.reshape(-1).to(device)
        out = model(x)
        r = 0
        if(out>0.5):
            r = 1
        print(i, r)
