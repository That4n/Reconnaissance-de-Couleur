import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Liste des couleurs de base avec leurs valeurs BGR
COLORS = {
    "Rouge": (0, 0, 255),
    "Vert": (0, 255, 0),
    "Bleu": (255, 0, 0),
    "Jaune": (0, 255, 255),
    "Violet": (128, 0, 128),
    "Cyan": (255, 255, 0),
    "Blanc": (255, 255, 255),
    "Noir": (0, 0, 0),
    "Orange": (0, 165, 255)
}

# Préparer les données d'entraînement
X_train = np.array(list(COLORS.values())) / 255.0  # Normalisation
y_train = np.array([list(COLORS.keys())])

# Construire un modèle simple avec TensorFlow
model = Sequential([
    Dense(10, activation='relu', input_shape=(3,)),
    Dense(10, activation='relu'),
    Dense(len(COLORS), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle (simple, sur des valeurs fixes)
y_train_one_hot = np.eye(len(COLORS))  # One-hot encoding
model.fit(X_train, y_train_one_hot, epochs=100, verbose=0)

# Capture vidéo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Récupération de la couleur au centre de l'image
    height, width, _ = frame.shape
    center_pixel = frame[height // 2, width // 2]
    center_pixel = np.array(center_pixel[::-1]) / 255.0  # Conversion en RGB et normalisation

    # Prédiction de la couleur
    prediction = model.predict(center_pixel.reshape(1, -1))
    color_name = list(COLORS.keys())[np.argmax(prediction)]

    # Affichage de la couleur détectée
    cv2.putText(frame, f"Couleur: {color_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.circle(frame, (width // 2, height // 2), 10, (0, 0, 0), -1)

    cv2.imshow("Reconnaissance de couleur", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
