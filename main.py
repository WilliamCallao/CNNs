import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando el dispositivo: {device}')

# Directorios del dataset
dataset_dir = os.path.join(os.getcwd(), "dataset")
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

# Verificar la existencia de los directorios
if not os.path.exists(train_dir):
    raise FileNotFoundError(f'El directorio de entrenamiento no existe: {train_dir}')
if not os.path.exists(test_dir):
    raise FileNotFoundError(f'El directorio de prueba no existe: {test_dir}')

# Transformaciones para los conjuntos de datos
transform = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
}

# Cargar los datasets
print('Cargando el conjunto de datos de entrenamiento...')
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform["train"])
print(f'Número de imágenes de entrenamiento: {len(train_dataset)}')

print('Cargando el conjunto de datos de prueba...')
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform["test"])
print(f'Número de imágenes de prueba: {len(test_dataset)}')

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Cargar el modelo
num_classes = len(train_dataset.classes)  # Detectar automáticamente las clases
print(f'Número de clases: {num_classes}')

print('Inicializando el modelo ResNet50...')
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Ajustar la última capa para el número de clases en tu dataset
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)
)

# Cargar los pesos del modelo de forma segura
model_path = 'model.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f'El archivo del modelo no existe: {model_path}')

print(f'Cargando los pesos del modelo desde {model_path}...')
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()
print('Modelo cargado y preparado para evaluación.')

# Función para medir la precisión
def evaluar_modelo(model, dataloader, nombre_conjunto):
    print(f'Evaluando el modelo en el conjunto de {nombre_conjunto}...')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Precisión en el conjunto de {nombre_conjunto}: {accuracy:.2f}%')
    return accuracy

# Evaluar el modelo
train_accuracy = evaluar_modelo(model, train_loader, 'entrenamiento')
test_accuracy = evaluar_modelo(model, test_loader, 'prueba')

print('Evaluación completada.')
