import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os


class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        
        # Filter and sort only color image files
        self.image_bases = [
            os.path.splitext(fname)[0].replace("_co", "") 
            for fname in os.listdir(images_dir) 
            if fname.endswith("_co.png")
        ]
        self.image_bases.sort()

    def __len__(self):
        return len(self.image_bases)

    def __getitem__(self, idx):
        # Load only color image
        base_name = self.image_bases[idx]
        color_image_path = os.path.join(self.images_dir, f"{base_name}_co.png")
        image = Image.open(color_image_path).convert("RGB")

        # Load corresponding annotations
        annotation_path = os.path.join(self.annotations_dir, f"{base_name}.txt")
        boxes, labels = self.parse_annotation(annotation_path)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Create the target dictionary
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return image, target

    def parse_annotation(self, annotation_path):
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

                # Validate bounding box
                if x_max > x_min and y_max > y_min:  # Ensure valid dimensions
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(parts[4]))  # Use the appropriate column for the class

        return boxes, labels

def get_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_model(images_dir, annotations_dir, output_model_path, num_classes=2, num_epochs=10, batch_size=2, lr=0.005):
    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor()])  # Convert PIL to Tensor
    dataset = VehicleDataset(images_dir, annotations_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),  # Required for object detection datasets
    )

    # Model and optimizer
    model = get_model(num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in dataloader:
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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")


if __name__ == "__main__":
    # Paths to your dataset
    images_dir = "/home/fabioski01/GEOINT_files/Vehicules512"
    annotations_dir = "/home/fabioski01/GEOINT_files/Annotations512"

    # Path to save the trained model
    output_model_path = "fasterrcnn_vehicle_detector.pth"

    # Train the model
    train_model(images_dir, annotations_dir, output_model_path, num_classes=2, num_epochs=10)
