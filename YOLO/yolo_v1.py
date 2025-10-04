from ultralytics import YOLO 
import cv2
from collections import defaultdict
import numpy as np

cap = cv2.VideoCapture(0)

# Ajuste da câmera para 640x640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

model = YOLO("runs\\detect\\train4\\weights\\best.pt")

track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = False    

# Paleta de cores (BGR no OpenCV) - você pode adicionar mais cores conforme o nº de classes
class_colors = {
    0: (255, 0, 0),      # Azul Conjunto ATPVII
    1: (0, 165, 255),    # Laranja Balaclava ATPVII
    2: (0, 255, 0),      # Verde Capacete
    3: (255, 255, 0),    # Ciano Abafador
    4: (255, 0, 255),    # Magenta Luva ATPVII
    5: (0, 0, 255),      # Vermelho Luva ATPVIV
}

while True:
    success, img = cap.read()

    if success:
        if seguir:
            results = model.track(img, persist=True, imgsz=640)
        else:
            results = (img, imgsz=640)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])   
                if conf >= 0.80:    # confiança mínima
                    cls = int(box.cls[0])   
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    label = f"{model.names[cls]} {conf:.2f}"

                    # Pega cor da classe (ou gera cor aleatória se não tiver definida)
                    color = class_colors.get(cls, (int(cls*40) % 255, int(cls*80) % 255, int(cls*120) % 255))

                    # Desenhar retângulo e texto
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if seguir and deixar_rastro:
                try:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                except:
                    pass

        cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("desligando")