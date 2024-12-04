import os
import shutil
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando el dispositivo: {device}')

# Directorios del dataset
dataset_dir = os.path.join(os.getcwd(), "dataset")
test_dir = os.path.join(dataset_dir, "test")
correct_output_dir = os.path.join(os.getcwd(), "imagenes_correctas")
incorrect_output_dir = os.path.join(os.getcwd(), "imagenes_incorrectas")

# Crear los directorios de salida si no existen
if not os.path.exists(correct_output_dir):
    os.makedirs(correct_output_dir)
    print(f'Carpeta creada: {correct_output_dir}')
if not os.path.exists(incorrect_output_dir):
    os.makedirs(incorrect_output_dir)
    print(f'Carpeta creada: {incorrect_output_dir}')

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

# Función para copiar imágenes clasificadas correctamente e incorrectamente
def copiar_imagenes_clasificadas(model, dataloader, dataset, correct_output_dir, incorrect_output_dir):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_predictions = (predicted == labels).cpu().numpy()
            for i in range(len(correct_predictions)):
                # Obtener la ruta completa de la imagen original
                img_path, _ = dataset.samples[total - len(correct_predictions) + i]
                if correct_predictions[i]:
                    # Directorio de la clase predicha (correcta)
                    class_dir = os.path.join(correct_output_dir, dataset.classes[predicted[i]])
                    if not os.path.exists(class_dir):
                        os.makedirs(class_dir)
                    # Copiar la imagen al directorio de la clase predicha
                    shutil.copy(img_path, class_dir)
                    print(f'Imagen correcta copiada: {img_path} a {class_dir}')
                    correct += 1
                else:
                    # Directorio de la clase real (incorrecta)
                    class_dir = os.path.join(incorrect_output_dir, dataset.classes[labels[i]])
                    if not os.path.exists(class_dir):
                        os.makedirs(class_dir)
                    # Copiar la imagen al directorio de la clase real
                    shutil.copy(img_path, class_dir)
                    print(f'Imagen incorrecta copiada: {img_path} a {class_dir}')
    
    accuracy = 100 * correct / total
    print(f'Precisión en el conjunto de prueba: {accuracy:.2f}%')

# Ejecutar la función para copiar las imágenes clasificadas correctamente e incorrectamente
copiar_imagenes_clasificadas(model, test_loader, test_dataset, correct_output_dir, incorrect_output_dir)
