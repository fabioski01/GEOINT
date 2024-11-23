import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    Carica il modello Faster R-CNN con pesi salvati.

    Args:
        model_path (str): Percorso del file con i pesi del modello.
        num_classes (int): Numero di classi, incluso lo sfondo.

    Returns:
        torch.nn.Module: Modello caricato.
    """
    print(f"Caricamento del modello da {model_path}...")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Imposta il modello in modalità inferenza
    print("Modello caricato e pronto per inferenza.")
    return model

def predict(model, image_path, device):
    """
    Effettua l'inferenza su una singola immagine.

    Args:
        model (torch.nn.Module): Modello addestrato.
        image_path (str): Percorso dell'immagine.
        device (torch.device): Dispositivo (CPU/GPU) per l'inferenza.

    Returns:
        list[dict]: Previsioni del modello (es. bounding boxes, classi, punteggi).
    """
    print(f"Caricamento immagine da {image_path}...")
    image = Image.open(image_path).convert("RGB")

    # Preprocessing
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Inference
    print("Eseguendo inferenza...")
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions[0]  # Ritorna i risultati per l'unica immagine

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_image_with_boxes(image, boxes, labels, scores, class_names, threshold=0.5):
    """
    Plotta un'immagine con i bounding boxes.

    Args:
        image (PIL.Image.Image): L'immagine da visualizzare.
        boxes (Tensor): Bounding boxes.
        labels (Tensor): Etichette delle classi.
        scores (Tensor): Confidenze delle predizioni.
        class_names (dict): Mappa id_classe -> nome_classe.
        threshold (float): Soglia per visualizzare i boxes.
    """
    # Converti immagine PIL in array NumPy per Matplotlib
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Plotta l'immagine
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Se boxes, labels o scores sono tensori 0-dimensionali, trattali come singolo elemento
    if boxes.dim() == 0:
        boxes = boxes.unsqueeze(0)
        labels = labels.unsqueeze(0)
        scores = scores.unsqueeze(0)

    # Assicurati che siano almeno 1D per iterare su di essi
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
        labels = labels.unsqueeze(0)
        scores = scores.unsqueeze(0)

    # Filtra le predizioni in base al threshold
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Converte il punteggio da tensore 0-dim a float, se necessario
        if isinstance(score, torch.Tensor) and score.dim() == 0:
            score = score.item()

        if score >= threshold:
            xmin, ymin, xmax, ymax = box

            # Disegna il rettangolo
            rect = plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                edgecolor='red', facecolor='none', linewidth=2
            )
            ax.add_patch(rect)

            # Aggiungi etichetta e score
            class_name = class_names.get(label.item(), str(label.item()))
            ax.text(
                xmin, ymin - 5, f'{class_name}: {score:.2f}',
                color='red', fontsize=12, backgroundcolor='white'
            )

    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Parametri
    model_path = "fasterrcnn_vehicle_detector.pth"  # Modifica con il tuo percorso
    test_image_path = "test_dimeas_cars.png"  # Immagine su cui fare inferenza
    image = Image.open(test_image_path).convert("RGB")
    num_classes = 12  # Modifica con il numero di classi del tuo dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Carica il modello
    model = load_model(model_path, num_classes)
    model.to(device)

    # Fai inferenza su un'immagine
    predictions = predict(model, test_image_path, device)

    # Stampa i risultati
    print("Risultati dell'inferenza:")
    for box, label, score in zip(
        predictions["boxes"], predictions["labels"], predictions["scores"]
    ):
        print(f"Box: {box}, Label: {label}, Score: {score}")
        plot_image_with_boxes(image, box, label, scores=score, class_names={1: "1", 2: "2", 3: "3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10", 11:"11", 12:"12"},threshold=0.5)

    plot_image_with_boxes(image, predictions["boxes"], predictions["labels"], scores=predictions["scores"],
                          class_names={1:"car", 2:"trucks", 4: "tractors", 5: "camping cars", 9: "vans", 10: "others", 11: "pickup", 23: "boats" , 201: "Small Land Vehicles", 301: "Large land Vehicles"}, threshold=0.5)