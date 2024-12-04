import os
import json
import torch
import gradio as gr
import pandas as pd
from torchvision import transforms, models
from PIL import Image

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando el dispositivo: {device}')

# Cargar el modelo pre-entrenado
num_classes = 12
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, num_classes)
)
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()

# Clases del modelo
class_names = ['Clase1', 'Clase2', 'Clase3', 'Clase4', 'Clase5',
               'Clase6', 'Clase7', 'Clase8', 'Clase9', 'Clase10', 
               'Clase11', 'Clase12']

# Cargar información de afecciones desde el archivo JSON
with open('afeccion_info.json', 'r') as f:
    afeccion_info = json.load(f)

# Transformaciones para preprocesar la imagen de entrada
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Función para realizar la predicción
def predecir_imagen(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_class = class_names[predicted_idx]
    
    info = afeccion_info.get(predicted_class, {})
    descripcion = info.get('descripcion', 'No disponible.')
    sintomas = info.get('sintomas', 'No disponible.')
    tratamientos = info.get('tratamientos', 'No disponible.')
    
    historial_entry = {
        'imagen': image.name if hasattr(image, 'name') else 'imagen_subida',
        'clase_predicha': predicted_class,
        'confianza': f"{confidence.item()*100:.2f}%",
        'descripcion': descripcion,
        'sintomas': sintomas,
        'tratamientos': tratamientos
    }
    guardar_historial(historial_entry)
    
    resultado = f"**Clase Predicha:** {predicted_class}\n"
    resultado += f"**Confianza:** {confidence.item()*100:.2f}%\n\n"
    resultado += f"**Descripción:** {descripcion}\n\n"
    resultado += f"**Síntomas Comunes:** {sintomas}\n\n"
    resultado += f"**Posibles Tratamientos:** {tratamientos}\n\n"
    resultado += "**Recomendación:** Se sugiere consultar a un dermatólogo para una evaluación más detallada."
    
    return resultado

# Función para guardar el historial
def guardar_historial(entry):
    historial_file = 'historial.json'
    if os.path.exists(historial_file):
        with open(historial_file, 'r') as f:
            historial = json.load(f)
    else:
        historial = []
    historial.append(entry)
    with open(historial_file, 'w') as f:
        json.dump(historial, f, indent=4)

# Función para mostrar el historial
def mostrar_historial():
    historial_file = 'historial.json'
    if os.path.exists(historial_file):
        with open(historial_file, 'r') as f:
            historial = json.load(f)
        df = pd.DataFrame(historial)
        return df
    else:
        return pd.DataFrame(columns=['imagen', 'clase_predicha', 'confianza', 'descripcion', 'sintomas', 'tratamientos'])

# Construir la interfaz con Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Clasificación de Afecciones de la Piel")
    gr.Markdown("Sube una imagen de una zona de la piel para obtener una predicción sobre la posible afección.")
    with gr.Tabs():
        with gr.TabItem("Análisis"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type='pil', label="Sube una imagen de la piel")
                    analyze_button = gr.Button("Analizar")
                with gr.Column(scale=1):
                    output_text = gr.Textbox(label="Resultados", lines=15)
            analyze_button.click(fn=predecir_imagen, inputs=image_input, outputs=output_text)
        with gr.TabItem("Historial"):
            historial_button = gr.Button("Mostrar Historial")
            historial_output = gr.Dataframe()
            historial_button.click(fn=mostrar_historial, inputs=None, outputs=historial_output)

# Ejecutar la interfaz
demo.launch()
