import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from config import config
import argparse
import random

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resolution', nargs=2, metavar=("Width", "Height"), type=int, help="Fuerza la resolución de la camara.")
    parser.add_argument('-f', '--fps', type=int, help="Fuerza los FPS de la camara.")
    parser.add_argument('-b', '--buffer', type=int, help="Fuerza el tamaño del buffer.")
    parser.add_argument('--wsl', action="store_true", help="Ajuste preestablecido para la camara en WSL usando usbipd (480p30 - buffer = 2)")

    args = parser.parse_args()

    if args.wsl:
        args.resolution = (640, 480)
        args.fps = 30
        args.buffer = 2

    if args.resolution is not None:
        args.width = args.resolution[0]
        args.height = args.resolution[1]
    
    exit(game(args))

# L2 Norm entre la mano y el punto objetivo (azul) 
def calculate_distance(point1, point2) -> float:
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance

FORMAS = ["circulo", "rectangulo", "cuadrado", "triangulo", "rombo", "estrella"]

def puntos_forma(tipo, cx, cy, r):
    if tipo == "circulo":
        return cv2.ellipse2Poly((cx, cy), (r, r), 0, 0, 360, 5)
    if tipo == "rectangulo":
        pts = [(-1.5*r, -0.9*r), (1.5*r, -0.9*r), (1.5*r, 0.9*r), (-1.5*r, 0.9*r)]
    elif tipo == "cuadrado":
        pts = [(-r, -r), (r, -r), (r, r), (-r, r)]
    elif tipo == "triangulo":
        pts = [(0, -r), (r, 0.8*r), (-r, 0.8*r)]
    elif tipo == "rombo":
        pts = [(0, -r), (r, 0), (0, r), (-r, 0)]
    elif tipo == "estrella":
        pts = []
        for k in range(10):
            ang = -np.pi / 2 + k * np.pi / 5
            rad = r if k % 2 == 0 else r * 0.4
            pts.append((rad * np.cos(ang), rad * np.sin(ang)))
    return np.array([(cx + x, cy + y) for x, y in pts], dtype=np.int32)

def generar_silueta(w, h, tipo=None):
    tipo = tipo or random.choice(FORMAS)
    r = random.randint(int(h * 0.09), int(h * 0.14))
    margen = 15
    ext_x = int(r * 1.5) + margen 

    cx = random.randint(ext_x, w - ext_x)               
    cy = random.randint(h // 2 + r + margen, h - r - 40)  

    return {"tipo": tipo, "centro": (cx, cy), "radio": r,
            "puntos": puntos_forma(tipo, cx, cy, r)}

def dibujar_silueta(frame, silueta, color=(255, 255, 255)):
    pts = silueta["puntos"]
    cv2.polylines(frame, [pts], True, color, 3)

def dibujar_divisoria(frame, color=(200, 200, 200)):
    h, w = frame.shape[:2]
    y = h // 2

    cv2.line(frame, (0, y), (w, y), color, 2)

    cv2.putText(frame, "Zona de dibujo", (10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

def game(args):
    #Para lo de las manos
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path="models/gesture_recognizer.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    # Inicializar landmarker
    with GestureRecognizer.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)  # Abrir cámara
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if args.resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.fps:
            cap.set(cv2.CAP_PROP_FPS, args.fps)
        if args.buffer:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffer)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  # seguridad si la cámara no da fps
            fps = 30

        mp_hands = mp.solutions.hands

        frame_ms = int(1000 / fps)

        # Inicializar valores
        timestamp = sum_t = counter = final_counter = 0
        last_time=time.time()

        silueta = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Voltear frame horizontalmente
            frame = cv2.flip(frame, 1)

            h_f, w_f = frame.shape[:2]
            if silueta is None:
                silueta = generar_silueta(w_f, h_f)

            # Variables de tiempo
            now = time.time()
            t = now - last_time
            last_time = now
            sum_t += t

            # Convertir frame a formato MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            dibujar_divisoria(frame)
            dibujar_silueta(frame, silueta)

            # Ejecutar detección
            result = landmarker.recognize_for_video(mp_image, timestamp)
            timestamp += frame_ms
            if result.gestures:
                for i, gestos in enumerate(result.gestures):
                    mano = result.handedness[i][0].category_name
                    gesto = gestos[0]
                    print(f"{mano}: {gesto.category_name} ({gesto.score:.2f})")

                # Dibujar landmarks si los hay
                if result.hand_landmarks:
                    for person_landmarks in result.hand_landmarks:  # para mantener tu estructura
                        h, w, _ = frame.shape

                        # Dibuja líneas entre los landmarks
                        for connection in mp_hands.HAND_CONNECTIONS:
                            start_idx, end_idx = connection
                            start = person_landmarks[start_idx]
                            end = person_landmarks[end_idx]

                            x1, y1 = int(start.x * w), int(start.y * h)
                            x2, y2 = int(end.x * w), int(end.y * h)

                            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # Dibuja los puntos (landmarks)
                        for idx, landmark in enumerate(person_landmarks):
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                        # El 8 es la punta del índice y el 4 la del pulgar
                        landmarks_points = [(person_landmarks[8].x, person_landmarks[8].y)]
                        for landmark in landmarks_points:
                            h, w, _ = frame.shape
                            x, y = int(landmark[0] * w), int(landmark[1] * h)


            # Mostrar tiempo restante de juego y puntos obtenidos
            cv2.putText(frame, f'Time: {config.game_time:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostrar información de FPS, Resolución y tamaño del buffer
            cv2.putText(
                frame,
                f'{int(cap.get(cv2.CAP_PROP_FPS))} - {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} - {int(cap.get(cv2.CAP_PROP_BUFFERSIZE))}',
                (10, 470),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 255, 0),
                1
            )
            config.game_time -= t
            
            if config.game_time <= 0:
                frame = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
                cv2.putText(frame, f'Press Enter to continue ...', (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Press ESC to exit', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)

            else:
                final_counter = counter

            # Mostrar resultado
            cv2.imshow("Pose Landmarker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if config.game_time <= 0 and key == 13:
                config.game_time = 20
                silueta = None
                counter = 0
                sum_t = 0
                last_time = time.time()

            # Tecla ESC para salir
            if cv2.waitKey(1) & 0xFF == 27:
                sys.exit(0)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()