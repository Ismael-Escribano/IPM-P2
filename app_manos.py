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

def mascara_contorno(puntos, w, h, grosor=1):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(m, [puntos], True, 255, grosor)
    return m

def puntuar(silueta_pts, trazos, w, h, tolerancia=15):
    validos = [np.array(s, dtype=np.int32) for s in trazos if len(s) > 1]
    if not validos:
        return 0

    objetivo = mascara_contorno(silueta_pts, w, h)
    trazo = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(trazo, validos, False, 255, 1)

    dist_a_objetivo = cv2.distanceTransform(255 - objetivo, cv2.DIST_L2, 3)
    dist_a_trazo = cv2.distanceTransform(255 - trazo, cv2.DIST_L2, 3)

    d1 = dist_a_objetivo[trazo > 0]
    precision = np.mean(d1 <= tolerancia) if d1.size else 0.0
    d2 = dist_a_trazo[objetivo > 0]
    cobertura = np.mean(d2 <= tolerancia) if d2.size else 0.0

    if precision + cobertura == 0:
        return 0
    return round(100 * 2 * precision * cobertura / (precision + cobertura))

def bbox_centro(trazos):
    pts = np.array([p for s in trazos for p in s])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return (x0 + x1) / 2, (y0 + y1) / 2

def desplazar(trazos, dx, dy):
    return [[(x + dx, y + dy) for x, y in s] for s in trazos]

def dibujar_trazos(frame, trazos, offset=(0, 0), color=(0, 255, 255), grosor=3):
    for s in trazos:
        if len(s) > 1:
            pts = np.array(desplazar([s], *offset)[0], dtype=np.int32)
            cv2.polylines(frame, [pts], False, color, grosor)

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

        estado = "dibujando"
        trazos = []
        trazo_actual = None
        frames_sin_dedo = 0
        frames_pulgar = 0
        desplazamiento = (0, 0)
        t_anim = t_resultado = 0
        ultima_puntuacion = 0
        puntos_total = 0
        figuras_hechas=0
        key = 0
        frames_borrar = 0
        t_borrado = 0

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

            # --- Qué hace la mano ---
            punta = None
            pulgar_arriba = False
            pulgar_abajo = False
            for i, gestos in enumerate(result.gestures):
                nombre = gestos[0].category_name
                lm = result.hand_landmarks[i]
                if nombre == "Pointing_Up" and punta is None:
                    punta = (int(lm[8].x * w_f), int(lm[8].y * h_f))
                elif nombre == "Thumb_Up":
                    pulgar_arriba = True
                elif nombre == "Thumb_Down":
                    pulgar_abajo = True

            y_div = h_f // 2
            offset = (0, 0)

            # --- Máquina de estados ---
            if estado == "dibujando":
                if punta is not None and punta[1] < y_div:
                    frames_sin_dedo = 0
                    if trazo_actual is None:
                        trazo_actual = []
                        trazos.append(trazo_actual)
                    trazo_actual.append(punta)
                else:
                    frames_sin_dedo += 1
                    if frames_sin_dedo > 4:      # tolera parpadeos del gesto
                        trazo_actual = None      # el siguiente trazo será nuevo

                frames_pulgar = frames_pulgar + 1 if pulgar_arriba else 0
                terminar = frames_pulgar >= config.frames_para_terminar or key == 32  # espacio

                validos = [s for s in trazos if len(s) > 1]
                if terminar and validos:
                    cx, cy = bbox_centro(validos)
                    dx = silueta["centro"][0] - cx if config.alinear_x else 0
                    dy = silueta["centro"][1] - cy
                    desplazamiento = (dx, dy)
                    trazos = validos
                    trazo_actual = None
                    frames_pulgar = 0
                    t_anim = time.time()
                    estado = "bajando"

                frames_borrar = frames_borrar + 1 if pulgar_abajo else 0
                if frames_borrar >= config.frames_para_borrar:
                    trazos = []
                    trazo_actual = None
                    frames_sin_dedo = 0
                    frames_borrar = 0
                    t_borrado = time.time()
                if time.time() - t_borrado < 1.0:
                    cv2.putText(frame, "Dibujo borrado", (w_f // 2 - 100, h_f // 2 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

            elif estado == "bajando":
                raw = min((time.time() - t_anim) / config.duracion_anim, 1.0)
                p = 1 - (1 - raw) ** 3           # frena al llegar
                offset = (desplazamiento[0] * p, desplazamiento[1] * p)
                if raw >= 1.0:
                    finales = desplazar(trazos, *desplazamiento)
                    ultima_puntuacion = puntuar(silueta["puntos"], finales, w_f, h_f)
                    puntos_total += ultima_puntuacion
                    figuras_hechas += 1
                    t_resultado = time.time()
                    estado = "resultado"

            elif estado == "resultado":
                offset = desplazamiento
                cv2.putText(frame, f"Puntuacion: {ultima_puntuacion}",
                            (w_f // 2 - 130, h_f // 2 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if time.time() - t_resultado > 2.5:
                    silueta = None               # se genera otra al inicio del bucle
                    trazos = []
                    estado = "dibujando"

            # --- Dibujar trazos y punta del dedo ---
            dibujar_trazos(frame, trazos, offset)
            if punta is not None:
                color = (0, 255, 0) if punta[1] < y_div else (0, 0, 255)
                cv2.circle(frame, punta, 8, color, 2)

            # --- Landmarks de la mano ---
            for person_landmarks in result.hand_landmarks:
                for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
                    s, e = person_landmarks[start_idx], person_landmarks[end_idx]
                    cv2.line(frame, (int(s.x * w_f), int(s.y * h_f)),
                             (int(e.x * w_f), int(e.y * h_f)), (255, 0, 0), 2)
                for lm_pt in person_landmarks:
                    cv2.circle(frame, (int(lm_pt.x * w_f), int(lm_pt.y * h_f)), 3, (0, 255, 0), -1)

            # --- Textos de ayuda ---
            ayuda = {"dibujando": "Indice arriba = dibujar | Pulgar arriba = terminar | Pulgar abajo = borrar",
                     "bajando": "", "resultado": ""}[estado]
            cv2.putText(frame, ayuda, (10, h_f - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Puntos: {puntos_total}", (w_f - 170, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


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
            if estado=="dibujando":
                config.game_time -= t
            
            if config.game_time <= 0:
                frame = np.zeros((h_f, w_f, 3), dtype=np.uint8)

                def centrado(texto, y, escala, color, grosor=2):
                    (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
                    cv2.putText(frame, texto, ((w_f - tw) // 2, y),
                                cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor)

                media = puntos_total / figuras_hechas if figuras_hechas else 0

                centrado("Tiempo agotado", 90, 1.2, (0, 255, 0), 3)
                centrado(f"Puntuacion final: {puntos_total}", 170, 1.0, (0, 255, 255))
                centrado(f"Figuras completadas: {figuras_hechas}", 220, 0.8, (255, 255, 255))
                centrado(f"Media por figura: {media:.0f}", 260, 0.8, (255, 255, 255))
                centrado("Enter: jugar de nuevo", 340, 0.8, (0, 255, 0))
                centrado("ESC: salir", 380, 0.8, (255, 128, 0))

            # Mostrar resultado
            cv2.imshow("Pose Landmarker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if config.game_time <= 0 and key == 13:
                config.game_time = 120
                silueta = None
                trazos = []
                trazo_actual = None
                estado = "dibujando"
                puntos_total = 0
                counter = 0
                sum_t = 0
                last_time = time.time()


        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()