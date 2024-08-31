import matplotlib.pyplot as plt
# 假设以下是你在训练过程中记录的数据

train_losses = [0.8, 0.6, 0.4, 0.3, 0.2]  # 示例训练集损失率
val_losses = [0.9, 0.7, 0.5, 0.4, 0.3]  # 示例验证集损失率
train_accuracies = [60, 70, 80, 85, 90]  # 示例训练集准确率
val_accuracies = [58, 68, 78, 82, 88]  # 示例验证集准确率
plt.figure(figsize=(10, 5))

# 绘制训练和验证损失率
plt.subplot(1, 2, 1)  # 创建1行2列的子图，当前为第1个子图
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证准确率
plt.subplot(1, 2, 2)  # 当前为第2个子图
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

# 展示图像
plt.tight_layout()
plt.show()
