import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import numpy as np


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_train = x_train.reshape(-1, 28,28,1)
x_test = x_test.reshape(-1, 28,28,1)


y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)), 
    layers.Dense(256, activation='relu'),  
    layers.Dense(128, activation='relu'),                      
    layers.Dense(10, activation='softmax')                   
])
 

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')


predictions = model.predict(x_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test[:5], axis=1)
params = model.count_params()
print(f'Tong so tham so cua mo hinh FClayer: {params}')

