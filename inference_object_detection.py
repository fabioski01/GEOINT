"""
This script demonstrates the use of a Faster R-CNN model for object detection on images.
It includes functionalities to load a pre-trained model, modify its classification head,
perform inference on single images, and visualize predictions with bounding boxes.

The main steps are:
1. Load a Faster R-CNN model pre-trained on COCO and modify it for custom classes.
2. Load pre-trained weights for the modified model.
3. Perform inference on a specified image.
4. Display the image with bounding boxes and class labels.
"""

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def get_model(num_classes):
    """
    Load a pre-trained Faster R-CNN model and replace the classifier head.

    Args:
        num_classes (int): Number of classes (including background).

    Returns:
        torch.nn.Module: The modified Faster R-CNN model.
    """
    print(f"Loading model with {num_classes} classes...")
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("Model ready for training.")
    return model

def load_model(model_path, num_classes):
    """
    Load a Faster R-CNN model with saved weights.

    Args:
        model_path (str): Path to the file with the model weights.
        num_classes (int): Number of classes, including background.

    Returns:
        torch.nn.Module: Loaded model.
    """
    print(f"Loading the model from {model_path}...")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded and ready for inference.")
    return model

def predict(model, image_path, device):
    """
    Perform inference on a single image.

    Args:
        model (torch.nn.Module): Trained model.
        image_path (str): Path to the image.
        device (torch.device): Device (CPU/GPU) for inference.

    Returns:
        list[dict]: Model predictions (e.g., bounding boxes, classes, scores).
    """
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")

    # Preprocessing
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Inference
    print("Performing inference...")
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions[0]  # Return results for the single image

def plot_image_with_boxes(image, image_filename, boxes, labels, scores, class_names, threshold=0.5):
    """
    Plot an image with bounding boxes.

    Args:
        image (PIL.Image.Image): The image to display.
        image_filename (str): Filename of the image being analyzed.
        boxes (Tensor): Bounding boxes.
        labels (Tensor): Class labels.
        scores (Tensor): Prediction confidence scores.
        class_names (dict): Mapping of class ID to class name.
        threshold (float): Threshold to display boxes.
    """
    # Convert PIL image to NumPy array for Matplotlib
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Plot the image
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Handle 0-dimensional tensors as single elements
    if boxes.dim() == 0:
        boxes = boxes.unsqueeze(0)
        labels = labels.unsqueeze(0)
        scores = scores.unsqueeze(0)

    # Ensure tensors are at least 1D for iteration
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
        labels = labels.unsqueeze(0)
        scores = scores.unsqueeze(0)

    # Filter predictions based on the threshold
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Convert the score from 0-dimensional tensor to float if needed
        if isinstance(score, torch.Tensor) and score.dim() == 0:
            score = score.item()

        if score >= threshold:
            xmin, ymin, xmax, ymax = box

            # Draw the rectangle
            rect = plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                edgecolor='red', facecolor='none', linewidth=2
            )
            ax.add_patch(rect)

            # Add label and score
            class_name = class_names.get(label.item(), str(label.item()))
            ax.text(
                xmin, ymin - 5, f'{class_name}: {score:.2f}',
                color='red', fontsize=12, backgroundcolor='white'
            )

    plt.axis("off")
    plt.savefig(f'plots/output_detection_{image_filename}')  # Save the plot as a file
    plt.show()

if __name__ == "__main__":
    # Parameters
    model_path = "/home/fabioski01/GEOINT_files/fasterrcnn_vehicle_detector.pth"  # Update with your model path
    image_filename = 'test_dimeas_cars.png'  # PNG only!
    test_image_path = f"/home/fabioski01/GEOINT/images_to_analyze/{image_filename}" 
    image = Image.open(test_image_path).convert("RGB")
    num_classes = 12  # Update with the number of classes in your dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model
    model = load_model(model_path, num_classes)
    model.to(device)

    # Perform inference on an image
    predictions = predict(model, test_image_path, device)

    # Print the results
    print("Inference results:")
    for box, label, score in zip(
        predictions["boxes"], predictions["labels"], predictions["scores"]
    ):
        print(f"Box: {box}, Label: {label}, Score: {score}")

    # Visualize predictions
    class_mapping = {
        1: "car", 2: "trucks", 4: "tractors", 5: "camping cars", 
        9: "vans", 10: "others", 11: "pickup", 23: "boats", 
        201: "Small Land Vehicles", 301: "Large land Vehicles"
    }
    plot_image_with_boxes(
        image, image_filename, predictions["boxes"], predictions["labels"], 
        scores=predictions["scores"], class_names=class_mapping, threshold=0.5
    )