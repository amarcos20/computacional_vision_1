import cv2
import numpy as np
import tensorflow as tf
import os

# --- Configurações ---
MODEL_PATH = os.path.join('models', 'hand_gesture_model.h5')
IMAGE_TO_PREDICT_PATH = 'path/to/your/image.jpg' # <-- MUDE ISTO para o caminho da sua imagem
TARGET_SIZE = (100, 100) # O mesmo tamanho usado no treino

def preprocess_image(image_path, target_size):
    """
    Carrega e pré-processa uma única imagem para que seja compatível com o modelo.
    """
    # Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None

    # Redimensionar para o mesmo tamanho do treino
    img_resized = cv2.resize(img, target_size)

    # Converter para float e normalizar (mesma etapa do validation_test_datagen)
    img_normalized = img_resized / 255.0

    # Expandir as dimensões para criar um "lote" de 1 imagem: (100, 100, 3) -> (1, 100, 100, 3)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def main():
    """
    Função principal para carregar o modelo e fazer a previsão.
    """
    print("--- Script de Inferência para Reconhecimento de Gestos ---")

    # 1. Carregar o modelo treinado
    print(f"A carregar o modelo de: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    # 2. Pré-processar a imagem de entrada
    print(f"A pré-processar a imagem: {IMAGE_TO_PREDICT_PATH}")
    processed_image = preprocess_image(IMAGE_TO_PREDICT_PATH, TARGET_SIZE)

    if processed_image is None:
        return

    # 3. Fazer a previsão
    print("A fazer a previsão...")
    prediction_prob = model.predict(processed_image)
    
    # 4. Apresentar o resultado
    predicted_class = np.argmax(prediction_prob, axis=1)[0]
    confidence = np.max(prediction_prob) * 100
    
    print("\n--- Resultado da Previsão ---")
    print(f"O número previsto é: {predicted_class}")
    print(f"Confiança da previsão: {confidence:.2f}%")

    # Opcional: Mostrar a imagem original
    original_image = cv2.imread(IMAGE_TO_PREDICT_PATH)
    cv2.putText(original_image, f"Previsto: {predicted_class} ({confidence:.2f}%)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Imagem com Previsão", original_image)
    cv2.waitKey(0) # Espera que uma tecla seja pressionada
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()