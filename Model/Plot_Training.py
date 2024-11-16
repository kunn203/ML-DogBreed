from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Load mô hình đã được huấn luyện
model = load_model('D:\\Python\\ML\\DoAnML\\DogBreedModel.h5')

# Chuẩn bị tập dữ liệu kiểm tra (validation)
datagen = ImageDataGenerator(rescale=1./255)
validation_generator = datagen.flow_from_directory(
    'D:\\Python\\ML\\DoAnML\\Images',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Sử dụng 'validation' subset
)



# Dự đoán kết quả trên tập validation
y_true = validation_generator.classes  # Nhãn thực tế
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)  # Lấy chỉ số lớp có xác suất cao nhất

# Tính các chỉ số
precision = precision_score(y_true, y_pred_classes, average='weighted')  # average='weighted' tính toán cho tất cả các lớp
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# In kết quả
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

# Vẽ Confusion Matrix
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Đánh giá mô hình
loss, accuracy = model.evaluate(validation_generator)
print(f'Dộ chính xác trên tập dữ liệu kiểm tra: {accuracy * 100:.2f}%')

# Nếu bạn có lịch sử huấn luyện (training history)
# Vẽ biểu đồ accuracy và loss nếu có dữ liệu về quá trình huấn luyện


# Tải lịch sử huấn luyện từ file .npy
history = np.load('training_history.npy', allow_pickle=True).item()
# Vẽ biểu đồ
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Accuracy & Loss')
plt.ylabel('Accuracy / Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
