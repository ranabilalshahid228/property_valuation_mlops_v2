import joblib
from xgboost import XGBRegressor
from tensorflow.keras import models, layers
from sklearn.preprocessing import StandardScaler

def train_production_model(df):
    X = df.drop(columns=['target_price']) # Update with your column name
    y = df['target_price']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')

    # Model 1: XGBoost Specialist
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6)
    xgb.fit(X_scaled, y)
    joblib.dump(xgb, 'models/xgb_model.pkl')

    # Model 2: Neural Network Generalist
    dnn = models.Sequential([
        layers.Input(shape=(X_scaled.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    dnn.compile(optimizer='adam', loss='huber') # Robust to outliers
    dnn.fit(X_scaled, y, epochs=50, batch_size=256, verbose=0)
    dnn.save('models/dnn_model.h5')
