import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
import csv
import shutil
import time

FACES_DIR = "faces"
UNKNOWN_DIR = "unknown"
ATTENDANCE_FILE = "asistencia.csv"

# Asegurar carpetas necesarias
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)


def get_encoded_faces():
    """
    Carga y codifica todas las imágenes de rostros conocidos que están
    en la carpeta FACES_DIR.

    :return: Diccionario donde la clave es el nombre (nombre del archivo
        sin extensión) y el valor es la codificación del rostro.
    """
    encoded = {}
    for dirpath, dnames, fnames in os.walk(FACES_DIR):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = face_recognition.load_image_file(
                    os.path.join(FACES_DIR, f)
                )
                encodings = face_recognition.face_encodings(face)
                if encodings:
                    encoded[f.split(".")[0]] = encodings[0]
    return encoded


def marcar_asistencia(nombre):
    """
    Registra la asistencia de una persona en un archivo CSV si no ha sido
    registrada antes.

    :param nombre: Nombre de la persona a registrar.
    """
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Nombre", "Hora"])

    with open(ATTENDANCE_FILE, "r+", newline="") as f:
        data = f.read()
        if nombre not in data:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer = csv.writer(f)
            writer.writerow([nombre, now])
            print(f"📝 Asistencia registrada: {nombre} a las {now}")


desconocidos_guardados = {}


def guardar_desconocido(frame, box):
    """
    Guarda una imagen de un rostro desconocido en la carpeta UNKNOWN_DIR,
    evitando guardar múltiples imágenes del mismo rostro en un intervalo
    de 5 segundos.

    :param frame: Frame actual de la cámara.
    :param box: Tupla con la posición del rostro (top, right, bottom, left).
    """
    global desconocidos_guardados
    top, right, bottom, left = box
    rostro = frame[top:bottom, left:right]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    now = time.time()

    # Evitar guardar múltiples imágenes en bucle
    for prev_time in desconocidos_guardados.values():
        if now - prev_time < 5:
            return

    filename = f"{UNKNOWN_DIR}/unknown_{timestamp}.jpg"
    cv2.imwrite(filename, rostro)
    print(f"🔍 Rostro desconocido guardado: {filename}")
    desconocidos_guardados[filename] = now


def reconocimiento_en_vivo():
    """
    Inicia la cámara y realiza reconocimiento facial en vivo.
    Detecta rostros conocidos y desconocidos, marca asistencia y guarda
    imágenes de desconocidos.
    Presiona 'q' para salir del modo en vivo.
    """
    known_faces = get_encoded_faces()
    known_encodings = list(known_faces.values())
    known_names = list(known_faces.keys())
    asistidos = set()

    video_capture = cv2.VideoCapture(0)
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        for face_encoding, location in zip(face_encodings, face_locations):
            name = "Desconocido"

            if known_encodings:
                matches = face_recognition.compare_faces(
                    known_encodings, face_encoding
                )
                face_distances = face_recognition.face_distance(
                    known_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            top, right, bottom, left = [v * 4 for v in location]
            color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
            )
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (255, 255, 255),
                1,
            )

            if name != "Desconocido" and name not in asistidos:
                marcar_asistencia(name)
                asistidos.add(name)
            elif name == "Desconocido":
                guardar_desconocido(frame, (top, right, bottom, left))

        cv2.imshow("Reconocimiento Facial", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Saliendo del reconocimiento en vivo...")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def etiquetar_caras_desconocidas():
    """
    Permite asignar nombres a las imágenes guardadas de rostros desconocidos.
    Muestra cada imagen y espera input:
      - ESC para omitir la imagen
      - Q para salir del etiquetado y volver al menú principal
      - Cualquier otra tecla para proceder a ingresar el nombre por consola.
    """
    print("\n=== ETIQUETAR ROSTROS DESCONOCIDOS ===")
    print("Se mostrará cada imagen desconocida.")
    print("Pulsa ESC para omitir la imagen.")
    print("Pulsa Q para salir del etiquetado.")
    print("Pulsa cualquier otra tecla para ingresar el nombre y luego pulsa Intro.\n")

    archivos = [
        f for f in os.listdir(UNKNOWN_DIR) if f.endswith(".jpg") or f.endswith(".png")
    ]

    if not archivos:
        print("No hay imágenes desconocidas para etiquetar.")
        return

    for archivo in archivos:
        ruta = os.path.join(UNKNOWN_DIR, archivo)
        imagen = cv2.imread(ruta)
        if imagen is None:
            print(f"No se pudo cargar la imagen: {archivo}")
            continue
        cv2.imshow("Asignar nombre (presiona ESC para omitir, Q para salir)", imagen)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC para omitir
            print(f"Omitido: {archivo}")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            continue
        elif key == ord("q"):
            print("Saliendo del etiquetado...")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break

        nombre = input(f"Ingrese el nombre de la persona en {archivo}: ").strip()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        if nombre:
            nuevo_nombre = (
                f"{nombre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            destino = os.path.join(FACES_DIR, nuevo_nombre)
            shutil.move(ruta, destino)
            print(f"✔️ Imagen movida a faces/ como {nuevo_nombre}")
        else:
            print("⚠️ Nombre vacío. Imagen no movida.")


def menu_principal():
    """
    Muestra el menú principal y gestiona las opciones del usuario:
      1 - Iniciar reconocimiento facial en vivo
      2 - Etiquetar rostros desconocidos
      3 - Salir del programa
    """
    while True:
        print("\n=== MENÚ ===")
        print("1. Iniciar reconocimiento facial en vivo")
        print("2. Etiquetar rostros desconocidos")
        print("3. Salir")
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            reconocimiento_en_vivo()
        elif opcion == "2":
            etiquetar_caras_desconocidas()
        elif opcion == "3":
            print("¡Hasta pronto!")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")


menu_principal()
