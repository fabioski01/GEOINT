import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
import os
from PIL import Image
import xml.etree.ElementTree as ET

class VehicleDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading paired color and infrared images along with their
    bounding box annotations for vehicle detection.
    """

    def __init__(self, images_dir, annotations_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            images_dir (str): Path to the directory containing the images.
            annotations_dir (str): Path to the directory containing the annotation files.
            transform (callable, optional): A function/transform to apply to the images (e.g., resizing, normalization).
        """
        self.images_dir = images_dir  # Directory containing the images (color and infrared)
        self.annotations_dir = annotations_dir  # Directory containing the annotation files
        self.transform = transform  # Transform to apply to images (if any)
        
        # List of base filenames (without file extensions) of the annotation files
        # e.g., for annotation file '00000000.txt', the base name will be '00000000'
        self.image_bases = [os.path.splitext(fname)[0] for fname in os.listdir(annotations_dir) if fname.endswith('.txt')]

        # Print the number of annotation files found for debugging
        print(f"Found {len(self.image_bases)} annotation files in {annotations_dir}")

    def __len__(self):
        """
        Return the total number of samples in the dataset (i.e., the number of annotation files).

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.image_bases)

    def __getitem__(self, idx):
        """
        Retrieve the image pair (color and infrared) and annotations for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (color_image, ir_image, annotations) where:
                - color_image: The color image (PIL Image or Tensor after transform).
                - ir_image: The infrared image (PIL Image or Tensor after transform).
                - annotations: Dictionary containing bounding box coordinates ('boxes') and class labels ('labels').
        """
        base_name = self.image_bases[idx]

        color_image_path = os.path.join(self.images_dir, f"{base_name}_co.png")
        ir_image_path = os.path.join(self.images_dir, f"{base_name}_ir.png")

        color_image = Image.open(color_image_path)
        ir_image = Image.open(ir_image_path)

        txt_file = os.path.join(self.annotations_dir, f"{base_name}.txt")

        annotations = self.parse_annotation(txt_file)
        
        # Check if 'boxes' tensor is empty by checking its size
        if annotations['boxes'].size(0) == 0:  # If no bounding boxes exist
            print(f"Warning: No bounding boxes found in {txt_file}")

        if self.transform:
            color_image = self.transform(color_image)
            ir_image = self.transform(ir_image)

        return color_image, ir_image, annotations


    def parse_annotation(self, txt_file):
        """
        Parse the annotation file to extract bounding boxes and class labels.
        
        Args:
            txt_file (str): Path to the annotation file (.txt).
        
        Returns:
            dict: A dictionary with two keys:
                - 'boxes': A tensor containing bounding box coordinates (xmin, ymin, xmax, ymax).
                - 'labels': A list of integer labels (e.g., 0 for background, 1 for vehicle).
        """
        boxes = []  # List to store bounding box coordinates
        labels = []  # List to store corresponding class labels

        # Check if the annotation file exists, if not print a warning
        if not os.path.exists(txt_file):
            print(f"Warning: {txt_file} does not exist.")
            return {'boxes': torch.tensor([]), 'labels': []}

        # Open and read the annotation file
        with open(txt_file, 'r') as f:
            for line in f:
                # Split the line into fields based on spaces
                fields = line.strip().split()
                
                # Ensure the line has enough data (at least 5 fields for bbox and label)
                if len(fields) < 5:
                    print(f"Skipping invalid line: {line}")
                    continue

                # Extract the bounding box (x, y, width, height) and the class label
                x, y, w, h = float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3])
                label = int(fields[4])
                
                # Extract the corners of the bounding box (xmin, ymin, xmax, ymax)
                bbox_corners = list(map(int, fields[5:]))
                xmin, ymin = bbox_corners[0], bbox_corners[1]
                xmax, ymax = bbox_corners[2], bbox_corners[3]

                # Add the bounding box and label to the lists
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)  # Store the class label (e.g., 2 for vehicle)

        # Return a dictionary containing the bounding boxes and labels
        return {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': labels}

# # 2. Define Image Transformations (resizing, normalization)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images to a fixed size (224x224)
#     transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
# ])

# # 3. Initialize Dataset and DataLoader for training
# train_dataset = VehicleDataset(images_dir='/home/fabioski01/GEOINT/Vehicules512', annotations_dir='/home/fabioski01/GEOINT/Annotations512', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# # 4. Define a simple Neural Network model (using a pretrained ResNet-18 model for transfer learning)
# class VehicleRecognitionModel(nn.Module):
#     def __init__(self):
#         super(VehicleRecognitionModel, self).__init__()
        
#         # Load a pretrained ResNet-18 model from torchvision
#         self.model = models.resnet18(pretrained=True)
        
#         # Modify the final fully connected layer to output 2 classes (vehicle types)
#         self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Example: 2 classes (vehicle or not)
    
#     def forward(self, x):
#         return self.model(x)

# # 5. Initialize the model, loss function, and optimizer
# model = VehicleRecognitionModel()  # Create the model instance

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss is commonly used for classification tasks
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with a learning rate of 0.001

# # 6. Training Loop
# num_epochs = 5  # Set number of epochs
# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode
#     running_loss = 0.0  # Initialize running loss for the epoch
    
#     # Iterate over batches of data
#     for images, annotations in train_loader:
#         # Zero the gradients of the optimizer (to prevent accumulation)
#         optimizer.zero_grad()
        
#         # Forward pass: Get predictions from the model
#         outputs = model(images)  # Model outputs predictions (class probabilities)
        
#         # Assuming the annotations contain the correct labels (for simplicity, assuming labels are integers)
#         labels = annotations['labels']  # These would need to be mapped to integer labels (e.g., vehicle=0, not_vehicle=1)
        
#         # Calculate the loss
#         loss = criterion(outputs, labels)  # Compare predictions to true labels
        
#         # Backward pass: Compute gradients
#         loss.backward()
        
#         # Optimizer step: Update model weights based on gradients
#         optimizer.step()
        
#         # Update running loss
#         running_loss += loss.item()
    
#     # Print loss for the epoch
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# # 7. Evaluation (Optional: To test the model on unseen data, e.g., a validation set)
# model.eval()  # Set model to evaluation mode (turns off dropout, batchnorm, etc.)
# with torch.no_grad():  # Disable gradient calculation for evaluation
#     for images, annotations in train_loader:  # You would use a validation set here
#         outputs = model(images)  # Get predictions from the model
#         # Evaluate performance (e.g., compute accuracy, IoU for bounding boxes)
#         # For simplicity, this step is skipped here

# print("Training Complete!")

dataset = VehicleDataset(images_dir='/home/fabioski01/GEOINT/Vehicules512', annotations_dir='/home/fabioski01/GEOINT/Annotations512', transform=None)
sample = dataset[0]  # Get the first sample
color_image, ir_image, annotations = sample
print(color_image.size, ir_image.size, annotations)