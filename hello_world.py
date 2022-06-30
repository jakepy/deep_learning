from tensorflow import keras
import numpy as np

## Define Our Model
model = keras.Sequential([keras.layers.Dense(
    units=1,
    input_shape=[1]
)])

## Compile Model
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

## Example Model Values
Xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

## Train Model
model.fit(Xs, Ys, epochs=500)

## View Results of the Model
print(model.predict([10.0]))
