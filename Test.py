import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
import os

def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to handle varying lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Perform morphological operations to remove noise and fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours in the image
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, image

def get_defect_locations(contours):
    CX, CY, C = [], [], []
    for contour in contours:
        # Filter out very small contours
        if cv2.contourArea(contour) > 50:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                CX.append(cx)
                CY.append(cy)
                C.append((cx, cy))
    return CX, CY, C

def predict_defects(model, image_path, defect_locations):
    im = Image.open(image_path).convert('L')  # Open as grayscale
    classes = {
        0: "missing_hole",
        1: "mouse_bite",
        2: "open_circuit",
        3: "short",
        4: "spur",
        5: "spurious_copper"
    }
    pred, confidence = [], []
    for c in defect_locations:
        im1 = im.crop((c[0]-32, c[1]-32, c[0]+32, c[1]+32))
        im1 = im1.resize((64, 64))  # Resize to match your model's input size
        im1 = np.array(im1)
        im1 = np.expand_dims(im1, axis=2)
        im1 = np.expand_dims(im1, axis=0)
        im1 = im1 / 255.0  # Normalize the image data
        a = model.predict(im1, verbose=0, batch_size=1)
        pred.append(np.argmax(a))
        confidence.append(a)
    return pred, confidence, classes

def plot_results(image, CX, CY, pred, confidence, classes):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(CX, CY, c='r', s=40)
    for i, txt in enumerate(pred):
        plt.annotate(f"{classes[txt]}\n{confidence[i][0][txt]:.2f}", (CX[i], CY[i]), color='r', fontsize=8)
    plt.title("PCB Defect Detection Results")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    model_path = 'D:\\PCB Defect Classification\\Model\\Model.h5'
    test_folder = 'D:\\PCB Defect Classification\\Test'

    model = load_model(model_path)

    for image_name in os.listdir(test_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_path = os.path.join(test_folder, image_name)
            
            print(f"Processing {image_name}")
            contours, image = process_image(test_path)
            CX, CY, C = get_defect_locations(contours)
            pred, confidence, classes = predict_defects(model, test_path, C)
            plot_results(image, CX, CY, pred, confidence, classes)

if __name__ == "__main__":
    main()