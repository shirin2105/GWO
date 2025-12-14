"""
CNN cơ bản cho nhận diện chữ số viết tay
Sử dụng hyperparameters khởi tạo ngẫu nhiên
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

class BasicCNN:
    def __init__(self, learning_rate=None, num_filters=None, random_seed=42):
        """
        Khởi tạo CNN với hyperparameters ngẫu nhiên hoặc được chỉ định
        
        Args:
            learning_rate: Learning rate (nếu None, sẽ khởi tạo ngẫu nhiên trong khoảng [0.0001, 0.01])
            num_filters: Số lượng filters cho conv layer đầu tiên (nếu None, sẽ khởi tạo ngẫu nhiên trong khoảng [16, 64])
            random_seed: Seed cho random generator
        """
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Khởi tạo ngẫu nhiên hyperparameters nếu không được chỉ định
        if learning_rate is None:
            self.learning_rate = np.random.uniform(0.0001, 0.01)
        else:
            self.learning_rate = learning_rate
            
        if num_filters is None:
            self.num_filters = int(np.random.uniform(16, 64))
        else:
            self.num_filters = num_filters
            
        print(f"Khởi tạo CNN với:")
        print(f"  - Learning Rate: {self.learning_rate:.6f}")
        print(f"  - Số Filters: {self.num_filters}")
        
        self.model = None
        self.history = None
        
    def load_data(self, data_dir='archive'):
        """
        Load dữ liệu từ thư mục archive
        """
        print("Đang load dữ liệu...")
        X = []
        y = []
        
        for label in range(10):
            folder_path = os.path.join(data_dir, str(label))
            if not os.path.exists(folder_path):
                print(f"Cảnh báo: Không tìm thấy thư mục {folder_path}")
                continue
                
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            print(f"Đang load {len(image_files)} ảnh từ class {label}...")
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (28, 28))  # Resize về 28x28
                    X.append(img)
                    y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize về [0, 1]
        X = X.astype('float32') / 255.0
        
        # Reshape để thêm channel dimension
        X = X.reshape(-1, 28, 28, 1)
        
        print(f"Đã load {len(X)} ảnh")
        print(f"Shape của X: {X.shape}")
        print(f"Shape của y: {y.shape}")
        
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """
        Xây dựng kiến trúc CNN
        """
        model = keras.Sequential([
            # Conv Layer 1
            layers.Conv2D(self.num_filters, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Conv Layer 2
            layers.Conv2D(self.num_filters * 2, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("\nKiến trúc CNN:")
        model.summary()
        
    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
        """
        Train model
        """
        print(f"\nBắt đầu training với {epochs} epochs...")
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Đánh giá model
        """
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nKết quả trên tập test:")
        print(f"  - Loss: {test_loss:.4f}")
        print(f"  - Accuracy: {test_acc:.4f}")
        
        return test_loss, test_acc
    
    def plot_history(self, save_path='basic_cnn_history.png'):
        """
        Vẽ đồ thị loss và accuracy
        """
        if self.history is None:
            print("Chưa train model!")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_title(f'Model Loss\nLR={self.learning_rate:.6f}, Filters={self.num_filters}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu đồ thị tại {save_path}")
        plt.close()
    
    def save_model(self, save_path='basic_cnn_model.h5'):
        """
        Lưu model
        """
        self.model.save(save_path)
        print(f"Đã lưu model tại {save_path}")


def main():
    """
    Chạy thử nghiệm với CNN cơ bản
    """
    print("="*60)
    print("CNN CƠ BẢN - HYPERPARAMETERS NGẪU NHIÊN")
    print("="*60)
    
    # Khởi tạo model với hyperparameters ngẫu nhiên
    cnn = BasicCNN()
    
    # Load data
    X_train, X_test, y_train, y_test = cnn.load_data()
    
    # Build model
    cnn.build_model()
    
    # Train
    cnn.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=32)
    
    # Evaluate
    test_loss, test_acc = cnn.evaluate(X_test, y_test)
    
    # Plot history
    cnn.plot_history()
    
    # Save model
    cnn.save_model()
    
    # Lưu thông tin hyperparameters
    with open('basic_cnn_params.txt', 'w') as f:
        f.write(f"Learning Rate: {cnn.learning_rate}\n")
        f.write(f"Số Filters: {cnn.num_filters}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test Loss: {test_loss}\n")
    
    print("\n" + "="*60)
    print("HOÀN TẤT!")
    print("="*60)


if __name__ == "__main__":
    main()
