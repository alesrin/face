
# Sistema de Reconocimiento Facial con Etiquetado

## Objetivo
Este proyecto permite realizar reconocimiento facial en vivo usando la cámara web, registrar asistencia automáticamente y gestionar rostros desconocidos guardando sus imágenes para posteriormente etiquetarlas con nombres. 

## Herramientas Utilizadas
- **Python 3**
- **face_recognition**: Biblioteca para detección y reconocimiento facial basada en dlib.
- **OpenCV**: Para captura y manipulación de video e imágenes.
- **NumPy**: Para operaciones numéricas.
- **CSV**: Para registrar asistencia en un archivo.
- **shutil & os**: Para manejo de archivos y directorios.
- **datetime**: Para timestamps en imágenes y registros.

## Estructura del Programa
- `faces/`: Carpeta donde se almacenan las imágenes con rostros conocidos, etiquetados con nombres.
- `unknown/`: Carpeta donde se guardan imágenes de rostros desconocidos detectados en vivo.
- `asistencia.csv`: Archivo donde se registra la asistencia con nombre y hora.

## Funciones Principales
- **get_encoded_faces()**: Carga y codifica los rostros conocidos.
- **marcar_asistencia(nombre)**: Registra la asistencia en el CSV una sola vez por persona.
- **guardar_desconocido(frame, box)**: Guarda imágenes de rostros desconocidos, limitando la frecuencia para evitar guardar repetidamente la misma cara.
- **reconocimiento_en_vivo()**: Modo de reconocimiento facial en vivo con la cámara, dibuja cuadros y nombres, registra asistencia y guarda desconocidos.
- **etiquetar_caras_desconocidas()**: Permite etiquetar manualmente las imágenes de rostros desconocidos y moverlas a la carpeta de rostros conocidos.
- **menu_principal()**: Muestra un menú para elegir entre reconocimiento en vivo, etiquetado o salir.

## Uso
1. Ejecutar el programa.
2. Elegir la opción deseada en el menú:
   - `1` para iniciar reconocimiento facial en vivo.
   - `2` para etiquetar imágenes desconocidas.
   - `3` para salir.
3. Durante el reconocimiento, presiona `q` para salir.
4. Al etiquetar, presiona `ESC` para saltar una imagen o `q` para volver al menú principal.

---

Este proyecto es ideal para control de asistencia, seguridad o experimentación con reconocimiento facial.
