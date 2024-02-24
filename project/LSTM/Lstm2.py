import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error


train_input = pd.read_excel('data.xlsx')
confirmed_cases = train_input['newconfcase'].tolist()
search_freq = train_input['searchfrequency'].tolist()
confirmed5_cases = train_input['newconfcase5dayslater'].tolist()

trainset_list = list(zip(confirmed_cases, search_freq, confirmed5_cases))
data_array = np.array(trainset_list)

x_train = data_array[:357, :2]
y_train = data_array[:357, 2]


x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))

num_samples = 357
num_features = 2


model = Sequential()
model.add(LSTM(units=64, activation='tanh', input_shape=(1, num_features)))

for _ in range(5):
    model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=1, activation='linear'))


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])


model.summary()

# EarlyStopping 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=400, batch_size=10, validation_split=0.2,callbacks=[early_stopping])


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_input = pd.read_excel('data.xlsx')
confirmed_cases_test = test_input['newconfcase'].tolist()
search_freq_test = test_input['searchfrequency'].tolist()
confirmed5_cases_test = test_input['newconfcase5dayslater'].tolist()

testset_list = list(zip(confirmed_cases_test, search_freq_test, confirmed5_cases_test))
data_array_test = np.array(testset_list)

x_test = data_array_test[357:396, :2]
y_test = data_array_test[357:396, 2]


x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))


y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f'MAE: {mae}')
print(f'MAPE: {mape:.2f}%')
# 시각화
x_test0 = data_array[357:396, 0]
plt.plot(x_test0, label='Actual confcase')
plt.plot(y_test, label='Actual Confirmed5 Cases')
plt.plot(y_pred, label='Predicted Confirmed5 Cases')
plt.title('Predicted Confirmed5 Cases')
plt.xlabel('date')
plt.ylabel('Confirmed5 Cases')
plt.legend()
#plt.ylim([0, 900])
plt.show()
