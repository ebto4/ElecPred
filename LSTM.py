

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Optimisations GPU ---
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU memory growth enabled.")
else:
    print("Aucun GPU détecté.")

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled:", mixed_precision.global_policy())

# --- Paramètres ---
sequence_length = 168  # 1 semaine d'historique (168 heures)
horizon = 168          # Prédire 1 semaine (168 heures) dans le futur

# --- 1. Charger et préparer les données ---
df = pd.read_csv("/content/ElecPred/Finland.csv", parse_dates=["Datetime (Local)"])
df = df.iloc[:, 3:]  # Suppression des colonnes inutiles
df.set_index("Datetime (Local)", inplace=True)
df = df.rename(columns={df.columns[0]: "Price"})
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Ajouter des features temporelles
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)
df['month'] = df.index.month

# --- 2. Normalisation des features ---
scaler_price = MinMaxScaler()
df['Price_norm'] = scaler_price.fit_transform(df[['Price']])

scaler_hour = MinMaxScaler()
df['hour_norm'] = scaler_hour.fit_transform(df[['hour']])

scaler_day = MinMaxScaler()
df['day_norm'] = scaler_day.fit_transform(df[['day_of_week']])

scaler_month = MinMaxScaler()
df['month_norm'] = scaler_month.fit_transform(df[['month']])

# Définir les colonnes de features à utiliser pour le modèle
feature_cols = ['Price_norm', 'hour_norm', 'day_norm', 'is_weekend', 'month_norm']

# --- 3. Création des séquences pour le LSTM ---
X, y = [], []
for i in range(sequence_length, len(df) - horizon):
    X.append(df.iloc[i - sequence_length:i][feature_cols].values)
    y.append(df.iloc[i:i + horizon]['Price_norm'].values)
    
X = np.array(X)  # forme : (n_samples, 168, n_features)
y = np.array(y)  # forme : (n_samples, 168)

print("Forme de X :", X.shape)
print("Forme de y :", y.shape)

# --- 4. Découpage en ensembles d'entraînement et de test ---
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- 5. Construction du modèle LSTM amélioré ---
model = Sequential()
# Couche 1 : Bidirectional LSTM avec 256 unités, retour des séquences
model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(sequence_length, len(feature_cols))))
model.add(Dropout(0.3))
# Couche 2 : Bidirectional LSTM avec 128 unités, retour des séquences
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
# Couche 3 : Bidirectional LSTM avec 64 unités, sans retour de séquences
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
# Couche Dense intermédiaire
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
# Couche de sortie : prédiction multi-sortie (168 valeurs)
model.add(Dense(horizon, dtype='float32'))
model.compile(optimizer='adam', loss='mse')
model.summary()

# --- Optionnel: Réduire le learning rate sur plateau ---
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# --- 6. Entraînement avec EarlyStopping et réduction du learning rate ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# --- 7. Prédiction sur le jeu de test ---
y_pred = model.predict(X_test)

# --- 8. Inverser la normalisation pour obtenir des valeurs réelles ---
y_pred_inv = scaler_price.inverse_transform(y_pred)
y_test_inv = scaler_price.inverse_transform(y_test)

# --- 9. Calculer les métriques globales ---
r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))

print("R² :", r2)
print("MAE :", mae)
print("RMSE :", rmse)

# --- 10. Visualiser la prédiction pour le premier échantillon du test ---
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv[0], label='Prix Réel')
plt.plot(y_pred_inv[0], label='Prix Prévu')
plt.title("Prédiction sur 1 semaine (168h) avec modèle LSTM amélioré\n(1er échantillon test)")
plt.xlabel("Heures dans le futur")
plt.ylabel("Prix (€/MWh)")
plt.legend()
plt.tight_layout()
plt.show()

# --- 11. Sauvegarder le modèle entraîné ---
model.save("/content/ElecPred/lstm_model_improved.h5")
print("Modèle sauvegardé sous '/content/ElecPred/lstm_model_improved.h5'")

