import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import numpy


class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        """
        Initialize the VehicleDataset.

        Args:
            images_dir (str): Path to the directory containing images.
            annotations_dir (str): Path to the directory containing annotation files.
            transform (callable, optional): Transform to apply to the images.
        """
        print("Initializing VehicleDataset...")
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        # Filter and sort only color image files (ignore infrared)
        print("Filtering and sorting color images...")
        self.image_bases = [
            os.path.splitext(fname)[0].replace("_co", "")
            for fname in os.listdir(images_dir)
            if fname.endswith("_co.png")
        ]
        self.image_bases.sort()
        print(f"Found {len(self.image_bases)} images.")

    def __len__(self):
        return len(self.image_bases)

    def __getitem__(self, idx):
        """
        Get a single data sample (image and its corresponding annotations).

        Args:
            idx (int): Index of the image to load.

        Returns:
            tuple: Image tensor and corresponding annotations.
        """
        print(f"Loading image and annotations for index {idx}...")
        base_name = self.image_bases[idx]

        # Load the color image
        color_image_path = os.path.join(self.images_dir, f"{base_name}_co.png")
        image = Image.open(color_image_path).convert("RGB")

        # Load the corresponding annotation file
        annotation_path = os.path.join(self.annotations_dir, f"{base_name}.txt")
        boxes, labels = self.parse_annotation(annotation_path)

        # Apply transformations (if provided)
        if self.transform:
            image = self.transform(image)

        # Create target dictionary
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        print(f"Image {base_name} loaded successfully with {len(boxes)} objects.")
        return image, target

    def parse_annotation(self, annotation_path):
        """
        Parse the annotation file to extract bounding boxes and labels.

        Args:
            annotation_path (str): Path to the annotation file.

        Returns:
            tuple: List of bounding boxes and corresponding labels.
        """
        print(f"Parsing annotations from {annotation_path}...")
        boxes = []
        labels = []
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.split()))

                # Extract corner coordinates
                corner_topleft_x, corner_topright_x, corner_bottomright_x, corner_bottomleft_x = parts[6:10]
                corner_topleft_y, corner_topright_y, corner_bottomright_y, corner_bottomleft_y = parts[10:14]

                # Compute bounding box
                x_min = min(corner_topleft_x, corner_topright_x, corner_bottomright_x, corner_bottomleft_x)
                y_min = min(corner_topleft_y, corner_topright_y, corner_bottomright_y, corner_bottomleft_y)
                x_max = max(corner_topleft_x, corner_topright_x, corner_bottomright_x, corner_bottomleft_x)
                y_max = max(corner_topleft_y, corner_topright_y, corner_bottomright_y, corner_bottomleft_y)

                # Validate bounding box dimensions
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(parts[4]))
                else:
                    print(f"Invalid box found: {[x_min, y_min, x_max, y_max]}. Skipping...")

        print(f"Parsed {len(boxes)} valid bounding boxes.")
        return boxes, labels


    def _find_classes(self):
        """
        Read all annotation files to find unique class labels.

        This method scans through all annotation files in the dataset directory,
        extracts the class labels from each file, and collects them into a set 
        to ensure uniqueness. The resulting set of class labels is sorted and returned.

        The class label is expected to be located in the 4th column (index 3) of each line
        in the annotation file. This column represents the object class for the bounding box.

        Example of annotation line format:
        290.348971  504.611640  3.012318    2       1      0        277                 303                 304                     279                 502                 498                 508                     511
        cx          cy          rot?        class?  class? class?   corner_topleft_x    corner_topright_x   corner_bottomright_x    corner_bottomleft_x corner_topleft_y    corner_topright_y   corner_bottomright_y    corner_bottomleft_y 

        Returns:
            list: A sorted list of unique class labels present in the dataset.
        """
        print("Finding unique classes in the dataset...")
        class_set = set()  # Store unique class labels
        for base_name in self.image_bases:
            # Path to annotation file
            annotation_path = os.path.join(self.annotations_dir, f"{base_name}.txt")
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.split()))
                    class_set.add(int(parts[3]))  # Extract class label at index 3

        unique_classes = sorted(class_set)
        print(f"Found {len(unique_classes)} unique classes: {unique_classes}")
        return unique_classes


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

def train_model(images_dir, annotations_dir, output_model_path, num_epochs=10, batch_size=2, lr=0.005):
    """
    Train the Faster R-CNN model.

    Args:
        images_dir (str): Path to the directory containing images.
        annotations_dir (str): Path to the directory containing annotations.
        output_model_path (str): Path to save the trained model.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
    """
    print("Preparing dataset and dataloader...")
    transform = transforms.Compose([transforms.ToTensor()])  # Convert PIL to Tensor
    dataset = VehicleDataset(images_dir, annotations_dir, transform)

    # Dynamically determine the number of classes (including background)
    num_classes = len(dataset._find_classes()) + 1
    print(f"Number of classes (including background): {num_classes}")

    # Initialize the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),  # Required for object detection datasets
    )

    print("Initializing model and optimizer...")
    model = get_model(num_classes)  # Pass dynamically determined num_classes
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    print("Starting training loop...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}...")
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            print(f"Batch {batch_idx + 1} loss: {losses.item():.4f}")

        print(f"Epoch {epoch + 1} completed. Average loss: {running_loss / len(dataloader):.4f}")

    print("Training completed. Saving the model...")
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    print("Starting script...")





    images_dir = "Vehicules512"
    annotations_dir = "Annotations512"
    output_model_path = "fasterrcnn_vehicle_detector.pth"

    train_model(images_dir, annotations_dir, output_model_path, num_epochs=10)
