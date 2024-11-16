import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.applications.resnet import ResNet50
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping, ModelCheckpoint

class DogBreedDataset:
    def __init__(self, data_path, target_size=(224, 224), batch_size=32):
        # Khởi tạo ImageDataGenerator với tỷ lệ rescale
        self.datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        self.target_size = target_size
        self.batch_size = batch_size
        self.data_path = data_path

    def get_train_generator(self):
        return self.datagen.flow_from_directory(
            self.data_path,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

    def get_validation_generator(self):
        return self.datagen.flow_from_directory(
            self.data_path,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )


class DogBreedModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=120, use_resnet=False, learning_rate=0.001):
        self.model = Sequential()

        # Sử dụng ResNet50 nếu được chỉ định
        if use_resnet:
            resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
            for layer in resnet.layers:
                layer.trainable = False  # Không train lại các lớp ResNet50
            self.model.add(resnet)
            self.model.add(Flatten())

        else:
            # Nếu không sử dụng ResNet50 thì dùng CNN cơ bản
            self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(128, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())

        # Thêm các lớp fully connected
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))  # Giảm overfitting
        self.model.add(Dense(num_classes, activation='softmax'))

        # Compile mô hình với learning rate được truyền vào
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        # Hiển thị cấu trúc mô hình
        return self.model.summary()

    def train(self, train_generator, validation_generator, epochs=50):
        # Tạo EarlyStopping và ModelCheckpoint
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_dog_breed_model.keras', monitor='val_accuracy', save_best_only=True)

        # Huấn luyện mô hình
        return self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=epochs,
            callbacks=[early_stopping, checkpoint]
        )

    def evaluate(self, validation_generator):
        # Đánh giá mô hình
        loss, accuracy = self.model.evaluate(validation_generator)
        print(f'Độ chính xác trên tập dữ liệu kiểm tra: {accuracy * 100:.2f}%')


class ModelSaver:
    def __init__(self, model):
        self.model = model

    def save(self, file_path):
        # Lưu mô hình dưới dạng file .h5
        self.model.save(file_path)


def main():
    # Đường dẫn tới dữ liệu hình ảnh
    data_path = 'D:\\Python\\ML\\DoAnML\\Images'

    # Tạo dataset cho huấn luyện và kiểm tra
    dataset = DogBreedDataset(data_path)
    train_generator = dataset.get_train_generator()
    validation_generator = dataset.get_validation_generator()

    # Khởi tạo mô hình nhận diện giống chó với ResNet50 và learning_rate = 0.0001
    dog_breed_model = DogBreedModel(input_shape=(224, 224, 3), num_classes=120, use_resnet=True, learning_rate=0.0001)

    # Hiển thị cấu trúc mô hình
    dog_breed_model.summary()

    # Huấn luyện mô hình
    training_model = dog_breed_model.train(train_generator, validation_generator, epochs=50)

    # Đánh giá mô hình
    dog_breed_model.evaluate(validation_generator)

    # Lưu mô hình đã huấn luyện
    saver = ModelSaver(dog_breed_model.model)
    saver.save('D:\\Python\\ML\\DoAnML\\Dog_Model_Training.h5')


if __name__ == "__main__":
    main()
