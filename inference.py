print('Starting script...')
import torch
print('torch imported')
from torchvision import transforms
print('torchvision imported')
from PIL import Image
print('PIL imported')
import matplotlib.pyplot as plt
print('matplotlib imported')
import matplotlib.patches as patches
print('patches imported')

from training_model import VehicleRecognitionModel
print('training_model imported')

print('loading')
# Load the trained model
model = VehicleRecognitionModel()  # Initialize the model architecture
model.load_state_dict(torch.load('vehicle_model.pth'))  # Load weights
model.eval()  # Set the model to evaluation mode
print('loading model complete')

# Define the image transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print('transformation complete')

# Load a satellite image
image_path = '/home/fabioski01/GEOINT_files/testing_img/screen1.png'
image = Image.open(image_path).convert('RGB')  # Open and ensure RGB format
input_image = transform(image).unsqueeze(0)  # Transform and add batc   h dimension
print('loading image complete')

# Perform inference
with torch.no_grad():
    output = model(input_image)  # Get predictions
print('inference complete')

# Assuming the output is a classification probability for 'car' or 'not car'
class_idx = torch.argmax(output, dim=1).item()  # Get predicted class index
confidence = torch.softmax(output, dim=1)[0][class_idx].item()  # Get confidence score

# Map class index to label
class_labels = {0: 'Not a Car', 1: 'Car'}  # Adjust labels as needed
predicted_label = class_labels[class_idx]

# Print the result
print(f"Predicted Class: {predicted_label}")
print(f"Confidence: {confidence:.2f}")

# Visualize the input image and prediction
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
plt.axis('off')
plt.show()
