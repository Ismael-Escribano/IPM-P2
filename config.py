from pathlib import Path
import pygame
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.widget import WidgetHandler

class Config:
    def __init__(self):
        self.model_path = Path(__file__).parent / 'models' / 'gesture_recognizer.task'
        self.padding = 100
        self.game_time = 120     # Duración del juego en segundos
        self.frames_para_terminar = 15
        self.duracion_anim = 1.0
        self.alinear_x = True
        self.tiempo_partida = 120
        self.frames_para_borrar = 15

        self._defaults = {
            "padding": self.padding,
            "game_time": self.game_time,
            "tiempo_partida": self.tiempo_partida,
            "frames_para_terminar": self.frames_para_terminar,
            "frames_para_borrar": self.frames_para_borrar,
            "duracion_anim": self.duracion_anim,
            "alinear_x": self.alinear_x,
        }
        
    def show_config_menu(self):
        pygame.init()
        screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("Configuración")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("arial", 20)

        # (atributo, etiqueta, texto visible, valores reales)
        opciones = [
            ("padding", "Padding:", ["50", "100", "150", "200"], [50, 100, 150, 200]),
            ("game_time", "Duración del juego (s):", ["60", "120", "180", "240"], [60, 120, 180, 240]),
            ("tiempo_partida", "Tiempo partida (s):", ["60", "120", "180", "240"], [60, 120, 180, 240]),
            ("frames_para_terminar", "Frames para terminar:", ["5", "10", "15", "30"], [5, 10, 15, 30]),
            ("frames_para_borrar", "Frames para borrar:", ["5", "10", "15", "30"], [5, 10, 15, 30]),
            ("duracion_anim", "Duración animación (s):", ["0.5", "1.0", "1.5", "2.0"], [0.5, 1.0, 1.5, 2.0]),
            ("alinear_x", "Alinear eje X:", ["Sí", "No"], [True, False]),
        ]

        self.__running = True
        
        def guardar():
            for attr, dd in dropdowns.items():
                valor = dd.getSelected()
                if valor is not None:
                    setattr(self, attr, valor)
            self.__running = False

        def cancelar():
            self.__running = False

        def default():
            for k, v in self._defaults.items():
                setattr(self, k, v)
            get_class_values()

        # Botones inferiores
        default_btn = Button(
            screen, 35, 520, 140, 45,
            text="Restablecer",
            fontSize=22,
            inactiveColour=(60, 160, 90),
            hoverColour=(80, 190, 110),
            textColour=(255, 255, 255),
            radius=8,
            onClick=default,
        )

        cancel_btn = Button(
            screen, 240, 520, 130, 45,
            text="Cancelar",
            fontSize=22,
            inactiveColour=(180, 70, 70),
            hoverColour=(210, 90, 90),
            textColour=(255, 255, 255),
            radius=8,
            onClick=cancelar,
        )

        save_btn = Button(
            screen, 440, 520, 130, 45,
            text="Guardar",
            fontSize=22,
            inactiveColour=(60, 160, 90),
            hoverColour=(80, 190, 110),
            textColour=(255, 255, 255),
            radius=8,
            onClick=guardar,
        )
        def get_class_values():
            global dropdowns, etiquetas
            dropdowns = {}
            etiquetas = []
            # Desplegables
            for i in range(len(opciones)):
                attr, texto, choices, values = opciones[i]
                y = 40 + i * 60
                actual = getattr(self, attr)
                nombre = ("Sí" if actual else "No") if isinstance(actual, bool) else str(actual)

                dropdowns[attr] = Dropdown(
                    screen, 340, y, 200, 38,
                    name=nombre,
                    choices=choices,
                    values=values,
                    borderRadius=5,
                    colour=(220, 220, 220),
                    direction="down",
                )
                etiquetas.append((texto, y + 8))

        get_class_values()

        while self.__running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.__running = False

            screen.fill((45, 45, 65))

            for texto, y in etiquetas:
                surf = font.render(texto, True, (255, 255, 255))
                screen.blit(surf, (50, y))
            
            pygame_widgets.update(events)
            pygame.display.update()
            pygame.display.flip()
            clock.tick(60)

        # Limpia los widgets para poder volver a abrir el menú más adelante sin duplicados
        for w in list(WidgetHandler.getWidgets()):
            WidgetHandler.removeWidget(w)

        pygame.quit()

config = Config()  # Instancia

if __name__ == '__main__':
    print(f'Configuración inicial: {config.__dict__}')
    config.show_config_menu()
    print(f'Configuración final: {config.__dict__}')