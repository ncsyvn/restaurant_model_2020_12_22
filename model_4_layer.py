import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import os

print(tf.__version__)

''' Importing the dataset '''
xlsx_file = 'Restaurant_Data.xlsx'
sheet = "Sheet1"
columns = ['Day', 'Month', 'Year', 'DayOfWeek', 'ModelId', 'Weather', 'Temperature', 'AmountOutput']
x_column = ['Day', 'Month', 'Year', 'DayOfWeek', 'ModelId', 'Weather', 'Temperature']
y_column = ['AmountOutput']
data = pd.read_excel(xlsx_file, sheet_name=sheet)
dataset = data[columns]
print(dataset)

''' Get x and y data'''
x = dataset[x_column]
y = dataset[y_column]

''' Splitting the dataset into the Training set and Test set '''
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop('AmountOutput')
test_labels = test_dataset.pop('AmountOutput')

print(train_dataset.shape)
print(test_dataset.shape)
print(train_labels.shape)
print(test_labels.shape)

''' Build model with keras '''


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='sigmoid'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


''' Build model '''
model = build_model()

''' Print model summary '''
print(model.summary())

''' Include the epoch in the file name (uses `str.format`) '''
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

''' Create a callback that saves the model's weights every 5 epochs '''
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=100)

EPOCHS = 500

'''............................... Train model .......................................'''
history = model.fit(
    train_dataset, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[cp_callback, tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

'''............................... Test Model.........................................'''
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

''' Print test predictions'''
test_predictions = model.predict(test_dataset).flatten()
# print(test_predictions)

mape = tf.keras.losses.MeanAbsolutePercentageError()
print("MAPE is: ", mape(test_labels, test_predictions).numpy()/100)


''' Save model '''
model.save('saved_model/model_4_layer')
# 0.512841