import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 1. Tải và tiền xử lý dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Chuẩn hóa giá trị pixel về khoảng [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Làm phẳng hình ảnh từ (28, 28) thành (784,)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Không cần one-hot encoding, giữ y_train và y_test là giá trị số thực (0-9)
# Ví dụ: nhãn 5 vẫn là 5.0
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# 2. Xây dựng mô hình ANN
model = models.Sequential([
    # Lớp đầu vào: 784 nơ-ron (từ 28x28 pixel)
    layers.Input(shape=(784,)),
    # Lớp ẩn 1: Fully connected, 128 nơ-ron, ReLU
    layers.Dense(128, activation='relu'),
    # Lớp ẩn 2: Fully connected, 64 nơ-ron, ReLU
    layers.Dense(64, activation='relu'),
    # Lớp đầu ra: 1 nơ-ron cho dự đoán giá trị số thực, không cần activation
    layers.Dense(1, activation=None)  # Hoặc activation='linear'
])

# 3. Biên dịch mô hình
model.compile(optimizer='adam',
              loss='mean_squared_error',  # MSE cho regression
              metrics=['mae'])  # Mean Absolute Error để đánh giá

# 4. In tóm tắt mô hình
model.summary()

# 5. Huấn luyện mô hình
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=32, 
                    validation_split=0.2)  # 20% dữ liệu huấn luyện dùng để kiểm tra

# 6. Đánh giá mô hình trên tập kiểm tra
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f'\nMất mát (MSE) trên tập kiểm tra: {test_loss:.4f}')
print(f'Sai số tuyệt đối trung bình (MAE) trên tập kiểm tra: {test_mae:.4f}')

# 7. Dự đoán một vài ví dụ
predictions = model.predict(x_test[:5])
print("\nDự đoán cho 5 mẫu đầu tiên:")
for i in range(5):
    predicted_value = predictions[i][0]
    true_value = y_test[i]
    print(f"Mẫu {i+1}: Dự đoán = {predicted_value:.2f}, Thực tế = {true_value:.1f}")

# 8. Vẽ biểu đồ mất mát và sai số
plt.figure(figsize=(12, 4))

# Biểu đồ mất mát (MSE)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Mất mát huấn luyện')
plt.plot(history.history['val_loss'], label='Mất mát kiểm tra')
plt.title('Mất mát (MSE) qua các epoch')
plt.xlabel('Epoch')
plt.ylabel('Mất mát (MSE)')
plt.legend()

# Biểu đồ sai số tuyệt đối trung bình (MAE)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='MAE huấn luyện')
plt.plot(history.history['val_mae'], label='MAE kiểm tra')
plt.title('Sai số tuyệt đối trung bình (MAE) qua các epoch')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()