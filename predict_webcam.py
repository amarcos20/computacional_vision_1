import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp # Importamos o MediaPipe
import os
# --- Configurações ---
MODEL_PATH = os.path.join('models', 'hand_gesture_model.h5')
TARGET_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 75 # Limite de confiança para mostrar a previsão (em %)

# --- Inicialização do MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Queremos detetar apenas uma mão para este problema
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Carregar o nosso Modelo de Classificação ---
print("A carregar o modelo de classificação...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo carregado.")

# --- Iniciar a Webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Virar o frame horizontalmente para uma visualização tipo "espelho"
    frame = cv2.flip(frame, 1)
    
    # Converter o frame de BGR para RGB (o MediaPipe espera RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar o frame com o MediaPipe para detetar mãos
    results = hands.process(frame_rgb)

    # Se uma mão for detetada
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # --- Desenhar o esqueleto da mão (opcional, mas ótimo para debug) ---
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- Calcular o Bounding Box (retângulo) à volta da mão ---
            h, w, _ = frame.shape
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)
            
            # Adicionar uma margem de segurança ao bounding box
            margin = 20
            x_min = max(0, x_min - margin)
            x_max = min(w, x_max + margin)
            y_min = max(0, y_min - margin)
            y_max = min(h, y_max + margin)

            # --- Recortar a imagem da mão (ROI) ---
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Apenas processar se a ROI for válida
            if hand_roi.size > 0:
                # --- Pré-processar a ROI para o nosso modelo ---
                img_resized = cv2.resize(hand_roi, TARGET_SIZE)
                img_normalized = img_resized / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)

                # --- Fazer a previsão com o nosso modelo ---
                prediction_prob = model.predict(img_batch)
                predicted_class = np.argmax(prediction_prob, axis=1)[0]
                confidence = np.max(prediction_prob) * 100
                
                # --- Desenhar o resultado no frame ---
                # Desenhar o retângulo à volta da mão detetada
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Apenas mostrar o texto se a confiança for alta
                if confidence > CONFIDENCE_THRESHOLD:
                    text = f"Previsto: {predicted_class} ({confidence:.2f}%)"
                    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar o frame final
    cv2.imshow('Reconhecimento de Gestos com MediaPipe - Pressione Q', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libertar tudo
hands.close()
cap.release()
cv2.destroyAllWindows()