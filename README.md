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
python app.py
```
