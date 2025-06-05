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

# Cargar rostros conocidos
def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk(FACES_DIR):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = face_recognition.load_image_file(os.path.join(FACES_DIR, f))
                encodings = face_recognition.face_encodings(face)
                if encodings:
                    encoded[f.split(".")[0]] = encodings[0]
    return encoded

# Registrar asistencia una sola vez
def marcar_asistencia(nombre):
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

# Guardar cara desconocida
desconocidos_guardados = {}
def guardar_desconocido(frame, box):
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

# Modo de reconocimiento facial en vivo
def reconocimiento_en_vivo():
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
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, location in zip(face_encodings, face_locations):
            name = "Desconocido"

            if known_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            top, right, bottom, left = [v * 4 for v in location]
            color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            if name != "Desconocido" and name not in asistidos:
                marcar_asistencia(name)
                asistidos.add(name)
            elif name == "Desconocido":
                guardar_desconocido(frame, (top, right, bottom, left))

        cv2.imshow('Reconocimiento Facial', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Saliendo del reconocimiento en vivo...")
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Modo para asignar nombres a rostros desconocidos
def etiquetar_caras_desconocidas():
    archivos = [f for f in os.listdir(UNKNOWN_DIR) if f.endswith(".jpg") or f.endswith(".png")]

    if not archivos:
        print("No hay imágenes desconocidas para etiquetar.")
        return

    for archivo in archivos:
        ruta = os.path.join(UNKNOWN_DIR, archivo)
        imagen = cv2.imread(ruta)

        if imagen is None:
            print(f"⚠️ No se pudo leer la imagen: {archivo}. Saltando.")
            continue

        cv2.imshow("Asignar nombre (presiona ESC para omitir, Q para volver al menú)", imagen)
        key = cv2.waitKey(0)

        if key == 27:  # ESC para omitir
            print(f"Omitido: {archivo}")
            cv2.destroyAllWindows()
            continue
        elif key == ord('q') or key == ord('Q'):  # salir al menú
            print("🔙 Volviendo al menú principal.")
            cv2.destroyAllWindows()
            break

        cv2.destroyAllWindows()
        nombre = input(f"Ingrese el nombre de la persona en {archivo}: ").strip()

        if nombre:
            nuevo_nombre = f"{nombre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            destino = os.path.join(FACES_DIR, nuevo_nombre)
            shutil.move(ruta, destino)
            print(f"✔️ Imagen movida a faces/ como {nuevo_nombre}")
        else:
            print("⚠️ Nombre vacío. Imagen no movida.")


# Menú principal
def menu_principal():
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