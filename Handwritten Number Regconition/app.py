import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'        # Tắt thông báo oneDNN
from tensorflow import keras

# --- CẤU HÌNH ---

# --- CẤU HÌNH ---
MODEL_PATH = os.path.join('bestmodel.keras')
SEGMENTED_DIR = os.path.join('output_digit') # Sửa ở đây nè ông

# 1. Load Model
if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file model tại {MODEL_PATH}")
    exit()

model = keras.models.load_model(MODEL_PATH)
print("--- Model Loaded Successfully ---")

def preprocess_for_mnist(image_path):
    # Đọc ảnh xám
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # MNIST chuẩn là chữ TRẮNG nền ĐEN. 
    # File segmentation của ông đã làm bước này rồi (THRESH_BINARY_INV), 
    # nên ở đây ta chỉ cần đảm bảo kích thước.

    # 1. Resize về 20x20 (giữ tỷ lệ để không bị méo chữ)
    h, w = img.shape
    if h > w:
        new_h, new_w = 20, int(20 * w / h)
    else:
        new_h, new_w = int(20 * h / w), 20
    img_resized = cv2.resize(img, (new_w, new_h))

    # 2. Tạo khung đen 28x28 và dán ảnh 20x20 vào giữa (Padding)
    # Bước này cực kỳ quan trọng vì model MNIST học trên ảnh có lề trống
    final_img = np.zeros((28, 28), dtype=np.uint8)
    pad_h = (28 - new_h) // 2
    pad_w = (28 - new_w) // 2
    final_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized

    # 3. Chuẩn hóa (Normalization)
    final_img = final_img.astype('float32') / 255.0
    final_img = np.expand_dims(final_img, axis=(0, -1)) # Shape thành (1, 28, 28, 1)
    
    return final_img

# --- CHẠY PREDICT ---
def main():
    # Lấy danh sách ảnh và sắp xếp theo tên (để đúng thứ tự số từ trái qua phải)
    digit_files = [f for f in os.listdir(SEGMENTED_DIR) if f.endswith(('.png', '.jpg'))]
    digit_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0])) # Sort theo số 'digit_0', 'digit_1'...

    print(f"Tìm thấy {len(digit_files)} chữ số. Đang dự đoán...\n")
    
    full_number = ""
    for file_name in digit_files:
        path = os.path.join(SEGMENTED_DIR, file_name)
        input_data = preprocess_for_mnist(path)
        
        if input_data is not None:
            prediction = model.predict(input_data, verbose=0)
            result = np.argmax(prediction)
            full_number += str(result)
            print(f"File {file_name} ---> Dự đoán: {result}")

    print("-" * 30)
    print(f"KẾT QUẢ CUỐI CÙNG: {full_number}")
    print("-" * 30)

if __name__ == "__main__":
    main()