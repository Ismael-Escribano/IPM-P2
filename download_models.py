import requests
import os
import tqdm

url = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
}

for key, value in url.items():
    print(f"Descargando {key} desde {value}")
    # Crear el directorio de destino si no existe
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    # Definir la ruta de destino
    archivo_destino = os.path.join(os.path.join(os.path.dirname(__file__), 'models'), key)

    if os.path.exists(archivo_destino):
        print(f"El archivo {archivo_destino} ya existe. Omitiendo descarga.")
        continue

    # Descargar el archivo
    response = requests.get(value, stream=True)
    if response.status_code == 200:
        with open(archivo_destino, 'wb') as f:
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
            for chunk in response.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)
        print("\nDescarga completada.")
    else:
        print(f"\nError al descargar: {response.status_code}")