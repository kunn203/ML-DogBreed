import cv2
import numpy as np
import keras
from keras.src.legacy.preprocessing import image

class DogBreedRecognizer:
    def __init__(self, model_path, class_names, input_size=(224, 224)):
        # Khởi tạo mô hình đã huấn luyện
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names
        self.input_size = input_size

    def preprocess_image(self, img):
        # Chuyển ảnh về kích thước input_size và chuẩn hóa
        img = cv2.resize(img, self.input_size)
        img = img.astype('float32') / 255.0  # Chuẩn hóa ảnh
        img = np.expand_dims(img, axis=0)    # Thêm chiều batch cho ảnh
        return img

    def predict(self, img):
        # Dự đoán giống chó từ hình ảnh đã được xử lý
        preprocessed_img = self.preprocess_image(img)
        prediction = self.model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return self.class_names[predicted_class]

class VideoCamera:
    def __init__(self, source=0):
        # Mở camera
        self.video = cv2.VideoCapture(source)

    def __del__(self):
        # Giải phóng tài nguyên khi kết thúc
        self.video.release()

    def get_frame(self):
        # Đọc frame từ camera
        ret, frame = self.video.read()
        return frame

def main():
    # Tên các giống chó
    dog_classes = ['Affenpinscher', 'Afghan Hound', 'African Hunting', 'Airedale', 'American Staffordshire Terrier', 
                   'Appenzeller', 'Australian Terrier', 'Basenji', 'Basset', 
                   'Beagle', 'Bedlington Terrier', 'Bernese Mountain', 'Black and Tan Coonhound', 
                   'Blenheim Spaniel', 'Bluetick', 'Boodhound', 'Border Collie', 'Border Terrier', 
                   'Borzoi', 'Boston Bull', 'Bouvier Des Flandres', 'Boxer', 'Brabancon Griffon', 
                   'Briard', 'Brittany Spaniel', 'Bull Mastiff', 'Cairn', 'Cardigan', 'Chesapeake Bay Retriever', 
                   'Chihuahua', 'Chow Chow', 'Clumber', 'Cocker Spaniel', 'Collie', 'Curly Coated Retriever', 'Dandie Dinmont', 
                   'Dhole', 'Dingo', 'Doberman', 'English Foxhound', 'English Setter', 'English Springer', 'EntleBucher', 'Eskimo', 
                   'Flat Coated Retriever', 'French Bulldog', 'German Shepherd', 'German Short Haired Pointer', 
                   'Giant Schnauzer', 'Golden Retriever', 'Gordon Setter', 'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain', 
                   'Groenendael', 'Ibizan Hound', 'Irish Setter', 'Irish Terrier','Irish Water Spaniel', 'Irish Wolfhound', 
                   'Italian Greyhound', 'Japanese Spaniel', 'Keeshond', 'Kelpie', 'Kerry Blue Terrier', 'Komondor', 'Kuvasz',
                   'Labrador Retriever', 'Lakeland Terrier', 'Leonberg', 'Lhasa', 'Malamute', 'Malinois', 'Maltese', 'Mexican Hairless', 
                   'Miniature Pinscher', 'Miniature Poodle', 'Miniature Schnauzer', 'Newfoundland','Norfolk Terrier', 'Norwegian Elkhound', 
                   'Norwich Terrier', 'Old English Sheepdog', 'Otterhound', 'Papillon', 'Pekinese', 'Pembroke', 'Pomeranian', 'Pug', 'Redbone', 
                   'Rhodesian Ridgeback', 'Rottweiler', 'Saint Bernard', 'Saluki', 'Samoyed', 'Schipperke', 'Scotch Terrier', 'Scottish Deerhound', 
                   'Sealyham Terrier', 'Shetland Sheepdog', 'Shih-Tzu', 'Siberian Husky', 'Silky Terrier', 'Soft-coated Wheaten Terrier', 
                   'Staffordshire Bull Terrier', 'Standard Poodle', 'Standard Schnauzer', 'Sussex Spaniel', 'Tibetan Mastiff', 'Tibetan Terrier', 
                   'Toy Poodle', 'Toy Terrier', 'Vizsla', 'Walker Hound', 'Weimaraner', 'Welsh Springer Spaniel', 'West Highland White Terrier', 
                   'Whippet', 'Wire Haired Fox Terrier', 'Yorkshire Terrier']  
                # Thêm các nhãn giống chó thực tế

    # Đường dẫn tới mô hình đã huấn luyện
    model_path = 'D:\\Python\\ML\\DoAnML\\DogBreedModel.h5'

    # Khởi tạo đối tượng nhận diện giống chó
    recognizer = DogBreedRecognizer(model_path=model_path, class_names=dog_classes)

    # Mở camera
    camera = VideoCamera()

    while True:
        # Lấy khung hình từ camera
        frame = camera.get_frame()

        # Dự đoán giống chó trong khung hình
        predicted_breed = recognizer.predict(frame)

        # Hiển thị tên giống chó lên khung hình
        cv2.putText(frame, f'Du Doan: {predicted_breed}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị khung hình
        cv2.imshow('Nhan Dien Giong Cho', frame)

        # Thoát chương trình nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Đóng cửa sổ hiển thị
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
