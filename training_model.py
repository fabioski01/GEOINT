import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
import os
from PIL import Image
import xml.etree.ElementTree as ET

# Dataset class to handle loading and processing of vehicle images and their annotations
class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None, max_objects=10):
        """
        Args:
            images_dir (str): Path to the directory containing images.
            annotations_dir (str): Path to the directory containing annotation files.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_objects (int): Maximum number of bounding boxes per image.
        """
        self.images_dir = images_dir  # Directory containing images
        self.annotations_dir = annotations_dir  # Directory containing annotation files
        self.transform = transform  # Transformation to apply on images (resize, etc.)
        self.max_objects = max_objects  # Max number of objects per image (for padding)
        
        # List all annotation files and sort them
        self.image_bases = [os.path.splitext(fname)[0] for fname in os.listdir(annotations_dir) if fname.endswith('.txt')]
        self.image_bases.sort()  # Ensure consistent indexing (if needed)

    def __len__(self):
        # Return the number of images (same as number of annotation files)
        return len(self.image_bases)

    def __getitem__(self, idx):
        # Get the base name (without extension) for the current image and annotation pair
        base_name = self.image_bases[idx]
        
        # Construct file paths for both color and infrared images
        color_image_path = os.path.join(self.images_dir, f"{base_name}_co.png")
        ir_image_path = os.path.join(self.images_dir, f"{base_name}_ir.png")

        # Load the images using PIL
        color_image = Image.open(color_image_path)
        ir_image = Image.open(ir_image_path)

        # Load the corresponding annotation file
        txt_file = os.path.join(self.annotations_dir, f"{base_name}.txt")
        annotations = self.parse_annotation(txt_file)  # Parse bounding box and label information

        # Padding annotations if the number of boxes is less than the max_objects
        num_boxes = annotations['boxes'].size(0)

        if num_boxes > self.max_objects:
            annotations['boxes'] = annotations['boxes'][:self.max_objects]
            annotations['labels'] = annotations['labels'][:self.max_objects]
        elif num_boxes < self.max_objects:
            padding = torch.zeros(self.max_objects - num_boxes, 4)
            annotations['boxes'] = torch.cat([annotations['boxes'], padding], dim=0)
            annotations['labels'] = torch.cat(
                [annotations['labels'], torch.zeros(self.max_objects - num_boxes, dtype=torch.int64)]
            )
        # Apply any specified transformations (e.g., resizing) to the images
        if self.transform:
            color_image = self.transform(color_image)
            ir_image = self.transform(ir_image)

        # Return the transformed images and their annotations
        return color_image, ir_image, annotations

    def parse_annotation(self, txt_file):
        """
        Parse a single annotation file and extract bounding boxes and labels.
        Each line in the annotation file corresponds to one object, with box coordinates and label.
        """
        boxes = []
        labels = []

        # Open and read the annotation file
        with open(txt_file, 'r') as f:
            for line in f:
                parts = list(map(float, line.split()))  # Convert line to a list of floats
                x1, y1, x2, y2 = parts[6:10]  # Extract bounding box coordinates
                label = int(parts[4])  # Extract object label (assuming it's at index 4)
                
                # Append the bounding box and label to respective lists
                boxes.append([x1, y1, x2, y2])
                labels.append(label)

        # Convert boxes and labels to tensors for PyTorch
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Return the parsed bounding boxes and labels
        return {'boxes': boxes, 'labels': labels}

# 2. Define Image Transformations (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size (224x224)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Define collate_fn for DataLoader
def collate_fn(batch):
    color_images = torch.stack([item[0] for item in batch])
    ir_images = torch.stack([item[1] for item in batch])
    annotations = {
        'boxes': torch.stack([item[2]['boxes'] for item in batch]),
        'labels': torch.stack([item[2]['labels'] for item in batch]),
    }
    return color_images, ir_images, annotations

# 3. Initialize Dataset and DataLoader for training
train_dataset = VehicleDataset(images_dir='/home/fabioski01/GEOINT/Vehicules512', annotations_dir='/home/fabioski01/GEOINT/Annotations512', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# 4. Define a simple Neural Network model (using a pretrained ResNet-18 model for transfer learning)
class VehicleRecognitionModel(nn.Module):
    def __init__(self):
        super(VehicleRecognitionModel, self).__init__()
        
        # Load a pretrained ResNet-18 model from torchvision
        self.model = models.resnet18(pretrained=True)
        
        # Modify the final fully connected layer to output 2 classes (vehicle types)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Example: 2 classes (vehicle or not)
    
    def forward(self, x):
        return self.model(x)

# 5. Initialize the model, loss function, and optimizer
model = VehicleRecognitionModel()  # Create the model instance

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss is commonly used for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with a learning rate of 0.001

# 6. Training Loop
num_epochs = 5  # Set number of epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for the epoch
    
    # Iterate over batches of data
    for color_images, ir_images, annotations in train_loader:
        # Zero the gradients of the optimizer (to prevent accumulation)
        optimizer.zero_grad()
        
        # Forward pass: Get predictions from the model
        outputs = model(color_images)  # Model outputs predictions (class probabilities)
        
        # Assuming the annotations contain the correct labels (for simplicity, assuming labels are integers)
        labels = torch.argmax(annotations['labels'], dim=1)   # These would need to be mapped to integer labels (e.g., vehicle=0, not_vehicle=1)
        
        # Calculate the loss
        loss = criterion(outputs, labels)  # Compare predictions to true labels
        
        # Backward pass: Compute gradients
        loss.backward()
        
        # Optimizer step: Update model weights based on gradients
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
    
    # Print loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# 7. Saving model
torch.save(model.state_dict(), 'vehicle_model.pth')

# 8. Evaluation (Optional: To test the model on unseen data, e.g., a validation set)
model.eval()  # Set the model to evaluation mode (disables dropout, batchnorm, etc.)
with torch.no_grad():  # Disable gradient calculations for evaluation
    for color_images, ir_images, annotations in train_loader:  # Use the train_loader for now, but a validation_loader would be better
        outputs = model(color_images)  # Use only color images for predictions
        # Add performance metrics calculation here if needed

# this ignores infrared images
# model.eval()
# with torch.no_grad():
#     for color_images, _, _ in train_loader:  # Ignore IR images and annotations
#         outputs = model(color_images)
#         # Add any additional performance metric calculations here, if necessary


print("Training Complete!")

# This loads a saved model
# model = VehicleRecognitionModel()
# model.load_state_dict(torch.load('vehicle_model.pth'))
# model.eval()  # Set the model to evaluation mode

# dataset = VehicleDataset(images_dir='/home/fabioski01/GEOINT/Vehicules512', annotations_dir='/home/fabioski01/GEOINT/Annotations512', transform=None)
# sample = dataset[0]  # Get the first sample
# color_image, ir_image, annotations = sample
# print(color_image.size, ir_image.size, annotations)