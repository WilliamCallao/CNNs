import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando el dispositivo: {device}')

# Directorios del dataset
test_dir = os.path.join(os.getcwd(), "dataset", "test")

# Transformaciones para el conjunto de datos de prueba
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Cargar el dataset de prueba
print('Cargando el conjunto de datos de prueba...')
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
print(f'Número de imágenes de prueba: {len(test_dataset)}')

# Crear DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Cargar el modelo
num_classes = len(test_dataset.classes)  # Detectar automáticamente las clases
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, num_classes)
)
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()

# Función para calcular la precisión del modelo
def calcular_precision(model, dataloader):
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
    print(f'Precisión en el conjunto de prueba: {accuracy:.2f}%')

# Ejecutar la función para calcular la precisión
calcular_precision(model, test_loader)
