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
confirmed5_cases = train_input['newconfcase5dayslater'].tolist()

trainset_list = list(zip(confirmed_cases, confirmed5_cases))

data_array = np.array(trainset_list)

prex_train = data_array[:357, 0]
x_train = np.empty((0,7))
for j in range(351):
  xk = np.array([])
  for i in range(7):
    i = i + j
    xk = np.append(xk, np.array(data_array[i, 0]))
  x_train = np.append(x_train,np.array([xk]),axis = 0)

y_train = data_array[6:357,1]


num_samples = 351
num_features = 1


model = Sequential()
model.add(Dense(units=64, input_dim=7, activation='linear'))

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
                    callbacks=[early_stopping],
                    verbose = 0
)

test_input = pd.read_excel('data.xlsx')
confirmed_cases_test = test_input['newconfcase'].tolist()
confirmed5_cases_test = test_input['newconfcase5dayslater'].tolist()



testset_list = list(zip(confirmed_cases_test, confirmed5_cases_test))

data_array_test = np.array(testset_list)



prex_test = data_array[357:396, 0]
x_test = np.empty((0,7))
for j in range(357,390):
  xk = np.array([])
  for i in range(7):
    i = i + j
    xk = np.append(xk, np.array(data_array[i, 0]))
  x_test = np.append(x_test,np.array([xk]),axis = 0)

y_test = data_array[363:396,1]

predictions = model.predict(x_test)
mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
print(f'MAPE: {mape:.2f}%') #예측 오차 분석

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE') #평균 절차 함수
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

y_pred = model.predict(x_test)
plt.plot(y_test, label='Actual Confirmed5 Cases')
plt.plot(y_pred, label='Predicted Confirmed5 Cases')
plt.title('Predicted Confirmed5 Cases')
plt.xlabel('date')
plt.ylabel('Confirmed5 Cases')
plt.legend()
plt.show()
