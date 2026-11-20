# IPM - Práctica 2 

### Autores:
#### Pablo Serna Soriano
#### Ismael Escribano Orts

---------------------------------------------------------

## Introducción

Explicación del juego y cómo funciona

## Prerequisitos

Tener instalado **Conda**, [instalar aquí](https://www.anaconda.com/docs/getting-started/miniconda/install).

## Requisitos

Crear un entorno de conda:
```bash
conda create -n IPM python=3.12
conda activate IPM
```

## Instalación

Instalar las dependencias necesarias de requirements.
```bash
pip install -r requirements.txt
```

## Descargar pesos

Script para poder descargar los pesos del modelo *Gesture Recognizer*:
```bash
python download_models.py
```

Si no descarga el modelo de esta manera se preguntará al usuario si lo quiere instalar al iniciar la app principal.



## Ejecución
```bash
python app_manos.py
```
La aplicación permite diferentes parametros:
- `-h` o `--help` $\rightarrow$ Muestra los parametros soportados
- `-r W H` o `--resolution W H` $\rightarrow$ Fuerza la resolución de la camara (si es posible) a "WxH"
- `-f N` o `--fps N` $\rightarrow$ Fuerza los FPS de la camara (si es posible) a "N"
- `-b N` o `--buffer N` $\rightarrow$ Fuerza el tamaño del buffer a "N"
- `--wsl` $\rightarrow$ Activa un ajuste preestablecido para el uso de una camara en WSL (Resolución 640x480 | FPS 30 | Buffer 2) 