import os

class Config:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'models/pose_landmarker_full.task')
        self.padding = 100
        self.game_time = 120     # Duración del juego en segundos
        self.frames_para_terminar = 15
        self.duracion_anim = 1.0
        self.alinear_x = False
        self.tiempo_partida = 120
        self.frames_para_borrar = 15

config = Config()  # Instancia única