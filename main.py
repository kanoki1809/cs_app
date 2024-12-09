import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import gdown
import numpy as np

# Định nghĩa danh sách các lớp
class_list = {
    '0': 'Speed limit (20km/h)',
    '1': 'Speed limit (30km/h)',
    '2': 'Speed limit (50km/h)',
    '3': 'Speed limit (60km/h)',
    '4': 'Speed limit (70km/h)',
    '5': 'Speed limit (80km/h)',
    '6': 'End of speed limit (80km/h)',
    '7': 'Speed limit (100km/h)',
    '8': 'Speed limit (120km/h)',
    '9': 'No passing',
    '10': 'No passing for vehicles over 3.5 metric tons',
    '11': 'Right-of-way at the next intersection',
    '12': 'Priority road',
    '13': 'Yield',
    '14': 'Stop',
    '15': 'No vehicles',
    '16': 'Vehicles over 3.5 metric tons prohibited',
    '17': 'No entry',
    '18': 'General caution',
    '19': 'Dangerous curve to the left',
    '20': 'Dangerous curve to the right',
    '21': 'Double curve',
    '22': 'Bumpy road',
    '23': 'Slippery road',
    '24': 'Road narrows on the right',
    '25': 'Road work',
    '26': 'Traffic signals',
    '27': 'Pedestrians',
    '28': 'Children crossing',
    '29': 'Bicycles crossing',
    '30': 'Beware of ice/snow',
    '31': 'Wild animals crossing',
    '32': 'End of all speed and passing limits',
    '33': 'Turn right ahead',
    '34': 'Turn left ahead',
    '35': 'Ahead only',
    '36': 'Go straight or right',
    '37': 'Go straight or left',
    '38': 'Keep right',
    '39': 'Keep left',
    '40': 'Roundabout mandatory',
    '41': 'End of no passing',
    '42': 'End of no passing by vehicles over 3.5 metric tons'
}

# Tiêu đề ứng dụng
st.title('DETECT TRAFFIC SIGNS')

# Load mô hình đã tải xuống
model = torch.load('vgg16_model.pth')
model.eval()

# Định nghĩa các phép biến đổi cần thiết cho hình ảnh
transform = transforms.Compose([
    transforms.Resize((50, 50)),  # Resize ảnh về kích thước chuẩn
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa ảnh
])

# Hiển thị giao diện tải ảnh
st.header('Image')
image = st.file_uploader('Choose an image:', type=['png', 'jpg', 'jpeg'])

# Xử lý ảnh và dự đoán
if image is not None:
    # Mở ảnh
    image = Image.open(image)
    st.image(image, caption='Test Image')

    if st.button('Predict'):
        # Tiền xử lý ảnh
        image_tensor = transform(image).unsqueeze(0)  # Thêm một chiều batch

        # Dự đoán
        with torch.no_grad():  # Không tính toán gradient trong quá trình đánh giá
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)  # Lấy lớp có xác suất cao nhất

        # Hiển thị kết quả
        st.header('Result')
        label = str(predicted.item())  # Lấy giá trị label
        st.text(class_list[label])  # Hiển thị nhãn kết quả (tên biển báo)
