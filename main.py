import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
import os
from PIL import Image
import xml.etree.ElementTree as ET

# 1. Dataset Class to load images and annotations (XML format)
class VehicleDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            annotations_dir (string): Directory with all the annotation XML files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir  # Directory containing images
        self.annotations_dir = annotations_dir  # Directory containing annotation files
        self.transform = transform  # Transform to apply to images

        # List of image file paths
        self.image_paths = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith('.jpg')]
        
    def __len__(self):
        """Return the total number of images in the dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Fetch an image and its corresponding annotation (bounding box and label)"""
        img_path = self.image_paths[idx]  # Get image path
        image = Image.open(img_path)  # Open image using PIL

        # Get corresponding annotation XML file path
        xml_file = os.path.join(self.annotations_dir, os.path.basename(img_path).replace('.jpg', '.xml'))
        
        # Parse the XML annotation to get the bounding boxes and labels
        annotations = self.parse_annotation(xml_file)
        
        # Apply any transformations like resizing, normalization, etc.
        if self.transform:
            image = self.transform(image)
        
        return image, annotations  # Return image and annotations
    
    def parse_annotation(self, xml_file):
        """Parse XML file to get the bounding boxes and labels"""
        tree = ET.parse(xml_file)  # Parse the XML
        root = tree.getroot()  # Get root of XML

        boxes = []
        labels = []
        
        # Iterate over all objects in the XML file (cars, trucks, etc.)
        for obj in root.findall('object'):
            label = obj.find('name').text  # Get the label of the object (vehicle type)
            
            # Get the bounding box coordinates (xmin, ymin, xmax, ymax)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Append the bounding box coordinates and label
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)  # You can later map labels to integers if needed
        
        # Return a dictionary containing the bounding boxes and labels
        return {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': labels}


# 2. Define Image Transformations (resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size (224x224)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# 3. Initialize Dataset and DataLoader for training
train_dataset = VehicleDataset(images_dir='/path/to/train/images', annotations_dir='/path/to/train/annotations', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

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
    for images, annotations in train_loader:
        # Zero the gradients of the optimizer (to prevent accumulation)
        optimizer.zero_grad()
        
        # Forward pass: Get predictions from the model
        outputs = model(images)  # Model outputs predictions (class probabilities)
        
        # Assuming the annotations contain the correct labels (for simplicity, assuming labels are integers)
        labels = annotations['labels']  # These would need to be mapped to integer labels (e.g., vehicle=0, not_vehicle=1)
        
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

# 7. Evaluation (Optional: To test the model on unseen data, e.g., a validation set)
model.eval()  # Set model to evaluation mode (turns off dropout, batchnorm, etc.)
with torch.no_grad():  # Disable gradient calculation for evaluation
    for images, annotations in train_loader:  # You would use a validation set here
        outputs = model(images)  # Get predictions from the model
        # Evaluate performance (e.g., compute accuracy, IoU for bounding boxes)
        # For simplicity, this step is skipped here

print("Training Complete!")
