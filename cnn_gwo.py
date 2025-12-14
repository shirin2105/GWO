"""
CNN với Grey Wolf Optimization (GWO) để tối ưu hóa hyperparameters
GWO sẽ tìm kiếm Learning Rate và Số lượng Filters tối ưu
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
import json


class GreyWolfOptimizer:
    """
    Grey Wolf Optimization algorithm để tối ưu hóa hyperparameters
    """
    def __init__(self, n_wolves=5, n_iterations=10, dim=2, lb=None, ub=None):
        """
        Args:
            n_wolves: Số lượng wolves (search agents)
            n_iterations: Số vòng lặp
            dim: Số chiều (2: learning_rate và num_filters)
            lb: Lower bounds cho mỗi dimension
            ub: Upper bounds cho mỗi dimension
        """
        self.n_wolves = n_wolves
        self.n_iterations = n_iterations
        self.dim = dim
        
        # Bounds mặc định
        if lb is None:
            # [learning_rate_min, num_filters_min]
            self.lb = np.array([0.0001, 16])
        else:
            self.lb = np.array(lb)
            
        if ub is None:
            # [learning_rate_max, num_filters_max]
            self.ub = np.array([0.01, 64])
        else:
            self.ub = np.array(ub)
        
        self.alpha_pos = None  # Best solution
        self.alpha_score = float('inf')
        
        self.beta_pos = None   # Second best
        self.beta_score = float('inf')
        
        self.delta_pos = None  # Third best
        self.delta_score = float('inf')
        
        self.convergence_curve = []
        self.all_scores = []
        
    def optimize(self, objective_function):
        """
        Thực hiện GWO optimization
        
        Args:
            objective_function: Hàm mục tiêu cần minimize (nhận vào position, trả về fitness score)
        """
        print("\n" + "="*60)
        print("BẮT ĐẦU GREY WOLF OPTIMIZATION")
        print("="*60)
        print(f"Số wolves: {self.n_wolves}")
        print(f"Số iterations: {self.n_iterations}")
        print(f"Bounds: LR=[{self.lb[0]}, {self.ub[0]}], Filters=[{self.lb[1]}, {self.ub[1]}]")
        
        # Khởi tạo vị trí của wolves ngẫu nhiên
        positions = np.random.uniform(0, 1, (self.n_wolves, self.dim))
        positions = self.lb + positions * (self.ub - self.lb)
        
        # Main loop
        for iteration in range(self.n_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.n_iterations} ---")
            
            iteration_scores = []
            
            # Đánh giá fitness cho mỗi wolf
            for i in range(self.n_wolves):
                # Đảm bảo position nằm trong bounds
                positions[i] = np.clip(positions[i], self.lb, self.ub)
                
                # Tính fitness
                fitness = objective_function(positions[i])
                iteration_scores.append(fitness)
                
                print(f"Wolf {i+1}: LR={positions[i][0]:.6f}, Filters={int(positions[i][1])}, Fitness={fitness:.4f}")
                
                # Update Alpha, Beta, Delta
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy() if self.beta_pos is not None else None
                    
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy() if self.alpha_pos is not None else None
                    
                    self.alpha_score = fitness
                    self.alpha_pos = positions[i].copy()
                    
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy() if self.beta_pos is not None else None
                    
                    self.beta_score = fitness
                    self.beta_pos = positions[i].copy()
                    
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = positions[i].copy()
            
            self.all_scores.append(iteration_scores)
            self.convergence_curve.append(self.alpha_score)
            
            print(f"\nBest so far (Alpha): LR={self.alpha_pos[0]:.6f}, Filters={int(self.alpha_pos[1])}, Score={self.alpha_score:.4f}")
            
            # Tính a (giảm tuyến tính từ 2 về 0)
            a = 2 - iteration * (2.0 / self.n_iterations)
            
            # Update vị trí của wolves
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    # Alpha
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - positions[i][j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # Beta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - positions[i][j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # Delta
                    r1 = np.random.random()
                    r2 = np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - positions[i][j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # Update position
                    positions[i][j] = (X1 + X2 + X3) / 3.0
        
        print("\n" + "="*60)
        print("KẾT THÚC OPTIMIZATION")
        print("="*60)
        print(f"Best Learning Rate: {self.alpha_pos[0]:.6f}")
        print(f"Best Số Filters: {int(self.alpha_pos[1])}")
        print(f"Best Score: {self.alpha_score:.4f}")
        
        return self.alpha_pos, self.alpha_score


class CNN_GWO:
    """
    CNN kết hợp với Grey Wolf Optimization
    """
    def __init__(self, n_wolves=5, n_iterations=10):
        self.n_wolves = n_wolves
        self.n_iterations = n_iterations
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.best_model = None
        self.gwo = None
        self.optimization_history = []
        
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
                    img = cv2.resize(img, (28, 28))
                    X.append(img)
                    y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize
        X = X.astype('float32') / 255.0
        X = X.reshape(-1, 28, 28, 1)
        
        print(f"Đã load {len(X)} ảnh")
        
        # Chia train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
    def build_and_evaluate_model(self, params):
        """
        Build và evaluate model với params cho trước
        Trả về validation loss (để minimize)
        
        Args:
            params: [learning_rate, num_filters]
        """
        learning_rate = params[0]
        num_filters = int(params[1])
        
        # Clear session để tránh memory leak
        tf.keras.backend.clear_session()
        
        # Build model
        model = keras.Sequential([
            layers.Conv2D(num_filters, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(num_filters * 2, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train với early stopping (ít epochs hơn để tiết kiệm thời gian)
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = model.fit(
            self.X_train, self.y_train,
            batch_size=32,
            epochs=15,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Lấy validation loss tốt nhất
        val_loss = min(history.history['val_loss'])
        val_acc = max(history.history['val_accuracy'])
        
        # Lưu lịch sử
        self.optimization_history.append({
            'learning_rate': learning_rate,
            'num_filters': num_filters,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        
        return val_loss
    
    def optimize_hyperparameters(self):
        """
        Sử dụng GWO để tìm hyperparameters tối ưu
        """
        self.gwo = GreyWolfOptimizer(
            n_wolves=self.n_wolves,
            n_iterations=self.n_iterations,
            dim=2,
            lb=[0.0001, 16],
            ub=[0.01, 64]
        )
        
        # Chạy optimization
        best_params, best_score = self.gwo.optimize(self.build_and_evaluate_model)
        
        return best_params, best_score
    
    def train_best_model(self, best_params, epochs=30):
        """
        Train model cuối cùng với best hyperparameters
        """
        learning_rate = best_params[0]
        num_filters = int(best_params[1])
        
        print("\n" + "="*60)
        print("TRAIN MODEL CUỐI CÙNG VỚI BEST HYPERPARAMETERS")
        print("="*60)
        print(f"Learning Rate: {learning_rate:.6f}")
        print(f"Số Filters: {num_filters}")
        
        tf.keras.backend.clear_session()
        
        # Build model
        model = keras.Sequential([
            layers.Conv2D(num_filters, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(num_filters * 2, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nKiến trúc CNN:")
        model.summary()
        
        # Train trên full training data
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            self.X_train, self.y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        self.best_model = model
        self.history = history
        
        # Evaluate trên test set
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"\nKết quả trên tập test:")
        print(f"  - Loss: {test_loss:.4f}")
        print(f"  - Accuracy: {test_acc:.4f}")
        
        return history, test_loss, test_acc
    
    def plot_convergence(self, save_path='gwo_convergence.png'):
        """
        Vẽ đồ thị convergence của GWO
        """
        if self.gwo is None:
            print("Chưa chạy optimization!")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.gwo.convergence_curve) + 1), 
                self.gwo.convergence_curve, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness (Validation Loss)', fontsize=12)
        plt.title('Grey Wolf Optimization - Convergence Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu convergence curve tại {save_path}")
        plt.close()
    
    def plot_training_history(self, save_path='gwo_cnn_history.png'):
        """
        Vẽ đồ thị training history
        """
        if not hasattr(self, 'history'):
            print("Chưa train model!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss (GWO Optimized)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Model Accuracy (GWO Optimized)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu training history tại {save_path}")
        plt.close()
    
    def save_results(self):
        """
        Lưu kết quả optimization
        """
        if self.gwo is None:
            return
        
        results = {
            'best_learning_rate': float(self.gwo.alpha_pos[0]),
            'best_num_filters': int(self.gwo.alpha_pos[1]),
            'best_validation_loss': float(self.gwo.alpha_score),
            'convergence_curve': [float(x) for x in self.gwo.convergence_curve],
            'optimization_history': self.optimization_history
        }
        
        with open('gwo_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("Đã lưu kết quả GWO tại gwo_results.json")
        
        # Lưu model
        if self.best_model is not None:
            self.best_model.save('gwo_cnn_model.h5')
            print("Đã lưu model tại gwo_cnn_model.h5")


def main():
    """
    Chạy CNN với GWO optimization
    """
    print("="*60)
    print("CNN VỚI GREY WOLF OPTIMIZATION")
    print("="*60)
    
    # Khởi tạo
    cnn_gwo = CNN_GWO(n_wolves=5, n_iterations=8)
    
    # Load data
    cnn_gwo.load_data()
    
    # Optimize hyperparameters
    best_params, best_score = cnn_gwo.optimize_hyperparameters()
    
    # Plot convergence
    cnn_gwo.plot_convergence()
    
    # Train model cuối cùng với best params
    history, test_loss, test_acc = cnn_gwo.train_best_model(best_params, epochs=30)
    
    # Plot training history
    cnn_gwo.plot_training_history()
    
    # Save results
    cnn_gwo.save_results()
    
    print("\n" + "="*60)
    print("HOÀN TẤT!")
    print("="*60)


if __name__ == "__main__":
    main()
