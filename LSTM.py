import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("Finland.csv")
df = df.iloc[:, 3:]
df.set_index("Datetime (Local)", inplace=True)

df = df.rename(columns={
    df.columns[0]: "Price"        # renomme dynamiquement la 2e colonne
})


# === Paramètres ===
sequence_length = 168  # 1 semaine d'historique (168 heures)
horizon = 168          # Prédire 1 semaine (168 heures) dans le futur

df.index = pd.to_datetime(df.index)

# Ajouter des features temporelles
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(float)
df['month'] = df.index.month

# Normaliser Price et les features temporelles (sauf is_weekend qui est déjà 0/1)
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

# === 2. Créer les séquences pour le LSTM ===
X, y = [], []
for i in range(sequence_length, len(df) - horizon):
    # Séquence d'entrée : on prend 'sequence_length' heures d'historique pour toutes les features
    X.append(df.iloc[i - sequence_length:i][feature_cols].values)
    # La cible est les 'horizon' heures suivantes, ici on prédit uniquement le Price (normalisé)
    y.append(df.iloc[i:i + horizon]['Price_norm'].values)

X = np.array(X)  # forme attendue : (n_samples, 168, n_features)
y = np.array(y)  # forme attendue : (n_samples, 168)

print("Forme de X :", X.shape)
print("Forme de y :", y.shape)

# === 3. Découper en ensembles d'entraînement et de test (en respectant l'ordre chronologique) ===
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# === 4. Construire un modèle LSTM avancé ===
model = Sequential()
# Couche 1 : Bidirectional LSTM avec retour des séquences
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(sequence_length, len(feature_cols))))
model.add(Dropout(0.2))
# Couche 2 : Seconde couche Bidirectional LSTM sans retour de séquence
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.2))
# Couche Dense intermédiaire
model.add(Dense(128, activation='relu'))
# Couche de sortie : prédiction multi-sortie (168 valeurs)
model.add(Dense(horizon))
model.compile(optimizer='adam', loss='mse')
model.summary()

# === 5. Entraîner le modèle avec EarlyStopping ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# === 6. Prédiction sur le jeu de test ===
y_pred = model.predict(X_test)

# === 7. Inverser la normalisation pour obtenir des valeurs réelles ===
# Note : On inverse uniquement pour la target (Price)
y_pred_inv = scaler_price.inverse_transform(y_pred)
y_test_inv = scaler_price.inverse_transform(y_test)

# === 8. Calculer les métriques (globales sur l'ensemble du test) ===
# On a des sorties multi-step, donc on a un tableau 2D ; on les aplatis pour un calcul global.
r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))

print("R² :", r2)
print("MAE :", mae)
print("RMSE :", rmse)

# === 9. Visualiser la prédiction pour le premier échantillon du test ===
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv[0], label='Prix Réel')
plt.plot(y_pred_inv[0], label='Prix Prévu')
plt.title("Prédiction sur 1 semaine (168h) avec features additionnelles\n(1er échantillon test)")
plt.xlabel("Heures dans le futur")
plt.ylabel("Prix (€/MWh)")
plt.legend()
plt.tight_layout()
plt.show()
