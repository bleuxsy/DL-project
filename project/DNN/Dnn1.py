import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

train_input = pd.read_excel('data.xlsx')
confirmed_cases = train_input['newconfcase'].tolist()
search_freq = train_input['searchfrequency'].tolist()
confirmed5_cases = train_input['newconfcase5dayslater'].tolist()

trainset_list = list(zip(confirmed_cases, search_freq, confirmed5_cases))

data_array = np.array(trainset_list)

x_train = data_array[:357, :2]
y_train = data_array[:357, 2]

num_samples = 357
num_features = 2
m = 357  # 데이터 개수
n = 14   # 입력값 개수
k = 64   # 첫 번째 은닉층의 유닛 수

W = np.random.randn(n, k)
b = np.random.randn(1, k)

X = x_train #크기가 m,n 인 행렬로 넣어야함
X_transpose = np.transpose(X)


Z_train = np.dot(X_transpose, W) + b

def linear(x):
    return x

A_train = linear(Z_train)


model = Sequential()
model.add(Dense(units=64, input_dim=n, activation='linear', weights=[W, b]))

for _ in range(4):
    model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

history = model.fit(x_train, y_train,
                    epochs=400,
                    batch_size=10,
                    validation_split=0.2,
                    callbacks=[early_stopping]
)

test_input = pd.read_excel('data.xlsx')
confirmed_cases_test = test_input['newconfcase'].tolist()
search_freq_test = test_input['searchfrequency'].tolist()
confirmed5_cases_test = test_input['newconfcase5dayslater'].tolist()

testset_list = list(zip(confirmed_cases_test, search_freq_test, confirmed5_cases_test))

data_array_test = np.array(testset_list)

x_test = data_array[357:396, :2]
y_test = data_array[357:396, 2]

predictions = model.predict(x_test)
mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
print(f'MAPE: {mape:.2f}%')

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

y_pred = model.predict(x_test)
x_test0 = data_array[357:396, 0]
plt.plot(x_test0, label='Actual confcase')
plt.plot(y_test, label='Actual Confirmed5 Cases')
plt.plot(y_pred, label='Predicted Confirmed5 Cases')
plt.title('Predicted Confirmed5 Cases')
plt.xlabel('date')
plt.ylabel('Confirmed5 Cases')
plt.legend()
plt.show()
