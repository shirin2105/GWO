"""
Script để chạy và so sánh 2 phương pháp:
1. CNN với hyperparameters ngẫu nhiên
2. CNN với Grey Wolf Optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from cnn_basic import BasicCNN
from cnn_gwo import CNN_GWO


def compare_methods():
    """
    So sánh 2 phương pháp và visualize kết quả
    """
    print("="*80)
    print("SO SÁNH CNN VỚI HYPERPARAMETERS NGẪU NHIÊN VS GWO OPTIMIZATION")
    print("="*80)
    
    results = {
        'basic_cnn': {},
        'gwo_cnn': {},
        'comparison': {}
    }
    
    # =====================================================
    # PHƯƠNG PHÁP 1: CNN với hyperparameters ngẫu nhiên
    # =====================================================
    print("\n" + "="*80)
    print("PHƯƠNG PHÁP 1: CNN VỚI HYPERPARAMETERS NGẪU NHIÊN")
    print("="*80)
    
    start_time = time.time()
    
    basic_cnn = BasicCNN()
    X_train, X_test, y_train, y_test = basic_cnn.load_data()
    basic_cnn.build_model()
    basic_cnn.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=32)
    basic_loss, basic_acc = basic_cnn.evaluate(X_test, y_test)
    basic_cnn.plot_history('results_basic_cnn_history.png')
    basic_cnn.save_model('results_basic_cnn_model.h5')
    
    basic_time = time.time() - start_time
    
    results['basic_cnn'] = {
        'learning_rate': basic_cnn.learning_rate,
        'num_filters': basic_cnn.num_filters,
        'test_accuracy': float(basic_acc),
        'test_loss': float(basic_loss),
        'training_time': basic_time
    }
    
    print(f"\nThời gian training: {basic_time:.2f} giây")
    
    # =====================================================
    # PHƯƠNG PHÁP 2: CNN với GWO
    # =====================================================
    print("\n" + "="*80)
    print("PHƯƠNG PHÁP 2: CNN VỚI GREY WOLF OPTIMIZATION")
    print("="*80)
    
    start_time = time.time()
    
    cnn_gwo = CNN_GWO(n_wolves=5, n_iterations=8)
    cnn_gwo.load_data()
    best_params, best_score = cnn_gwo.optimize_hyperparameters()
    cnn_gwo.plot_convergence('results_gwo_convergence.png')
    history, gwo_loss, gwo_acc = cnn_gwo.train_best_model(best_params, epochs=30)
    cnn_gwo.plot_training_history('results_gwo_cnn_history.png')
    cnn_gwo.save_results()
    
    gwo_time = time.time() - start_time
    
    results['gwo_cnn'] = {
        'learning_rate': float(best_params[0]),
        'num_filters': int(best_params[1]),
        'test_accuracy': float(gwo_acc),
        'test_loss': float(gwo_loss),
        'training_time': gwo_time,
        'optimization_score': float(best_score)
    }
    
    print(f"\nThời gian training (bao gồm optimization): {gwo_time:.2f} giây")
    
    # =====================================================
    # SO SÁNH KẾT QUẢ
    # =====================================================
    print("\n" + "="*80)
    print("KẾT QUẢ SO SÁNH")
    print("="*80)
    
    print("\n1. CNN với Hyperparameters Ngẫu Nhiên:")
    print(f"   - Learning Rate: {results['basic_cnn']['learning_rate']:.6f}")
    print(f"   - Số Filters: {results['basic_cnn']['num_filters']}")
    print(f"   - Test Accuracy: {results['basic_cnn']['test_accuracy']:.4f}")
    print(f"   - Test Loss: {results['basic_cnn']['test_loss']:.4f}")
    print(f"   - Thời gian: {results['basic_cnn']['training_time']:.2f}s")
    
    print("\n2. CNN với GWO Optimization:")
    print(f"   - Learning Rate: {results['gwo_cnn']['learning_rate']:.6f}")
    print(f"   - Số Filters: {results['gwo_cnn']['num_filters']}")
    print(f"   - Test Accuracy: {results['gwo_cnn']['test_accuracy']:.4f}")
    print(f"   - Test Loss: {results['gwo_cnn']['test_loss']:.4f}")
    print(f"   - Thời gian: {results['gwo_cnn']['training_time']:.2f}s")
    
    # Tính improvement
    acc_improvement = (results['gwo_cnn']['test_accuracy'] - results['basic_cnn']['test_accuracy']) * 100
    loss_improvement = ((results['basic_cnn']['test_loss'] - results['gwo_cnn']['test_loss']) / results['basic_cnn']['test_loss']) * 100
    
    results['comparison'] = {
        'accuracy_improvement_percent': float(acc_improvement),
        'loss_improvement_percent': float(loss_improvement),
        'time_difference': gwo_time - basic_time
    }
    
    print("\n3. So sánh:")
    print(f"   - Cải thiện Accuracy: {acc_improvement:+.2f}%")
    print(f"   - Cải thiện Loss: {loss_improvement:+.2f}%")
    print(f"   - Thời gian chênh lệch: {results['comparison']['time_difference']:+.2f}s")
    
    if results['gwo_cnn']['test_accuracy'] > results['basic_cnn']['test_accuracy']:
        print(f"\n   ✓ GWO cho kết quả TỐT HƠN với accuracy cao hơn {acc_improvement:.2f}%")
    else:
        print(f"\n   ✗ GWO cho kết quả kém hơn với accuracy thấp hơn {abs(acc_improvement):.2f}%")
    
    # =====================================================
    # VẼ BIỂU ĐỒ SO SÁNH
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. So sánh Accuracy
    ax1 = axes[0, 0]
    methods = ['Random\nHyperparams', 'GWO\nOptimized']
    accuracies = [results['basic_cnn']['test_accuracy'], results['gwo_cnn']['test_accuracy']]
    colors = ['#3498db', '#e74c3c']
    bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim([min(accuracies) - 0.05, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị lên bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. So sánh Loss
    ax2 = axes[0, 1]
    losses = [results['basic_cnn']['test_loss'], results['gwo_cnn']['test_loss']]
    bars2 = ax2.bar(methods, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. So sánh Hyperparameters
    ax3 = axes[1, 0]
    x = np.arange(2)
    width = 0.35
    
    lr_values = [results['basic_cnn']['learning_rate'] * 1000, 
                 results['gwo_cnn']['learning_rate'] * 1000]  # x1000 để dễ hiển thị
    filters_values = [results['basic_cnn']['num_filters'], 
                     results['gwo_cnn']['num_filters']]
    
    ax3_2 = ax3.twinx()
    
    bars3_1 = ax3.bar(x - width/2, lr_values, width, label='Learning Rate (×10⁻³)', 
                      color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars3_2 = ax3_2.bar(x + width/2, filters_values, width, label='Num Filters',
                       color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Learning Rate (×10⁻³)', fontsize=10, fontweight='bold', color='#3498db')
    ax3_2.set_ylabel('Number of Filters', fontsize=10, fontweight='bold', color='#e74c3c')
    ax3.set_title('Hyperparameters Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.tick_params(axis='y', labelcolor='#3498db')
    ax3_2.tick_params(axis='y', labelcolor='#e74c3c')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. So sánh Training Time
    ax4 = axes[1, 1]
    times = [results['basic_cnn']['training_time'], results['gwo_cnn']['training_time']]
    bars4 = ax4.bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=150, bbox_inches='tight')
    print("\nĐã lưu biểu đồ so sánh tại results_comparison.png")
    plt.close()
    
    # =====================================================
    # LƯU KẾT QUẢ
    # =====================================================
    results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open('comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("Đã lưu kết quả so sánh tại comparison_results.json")
    
    print("\n" + "="*80)
    print("HOÀN TẤT SO SÁNH!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = compare_methods()
