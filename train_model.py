import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import subprocess
import re

def get_wifi_features():
    """擷取環境特徵 (AP數量, 平均RSSI, 變異數)"""
    try:
        cmd = "sudo iw dev wlan0 scan"
        res = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')
        signals = [float(s) for s in re.findall(r'signal: (-\d+\.\d+) dBm', res)]
        if not signals: return [0.0, -90.0, 0.0]
        return [len(signals), np.mean(signals), np.std(signals)]
    except:
        return [0.0, -90.0, 0.0]

def build_autoencoder():
    """建構 3 -> 2 -> 3 的壓縮重建架構"""
    model = models.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(2, activation='relu'),
        layers.Dense(3, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    print("開始收集家中背景特徵 (預計 10 分鐘)...")
    data = []
    for _ in range(300): # 每 2 秒掃描一次，共 600 秒
        data.append(get_wifi_features())
        time.sleep(2)
        
    X_train = np.array(data)
    # 儲存家中的特徵分佈基準 (供後續正規化使用)
    np.save('home_scaler.npy', [np.mean(X_train, axis=0), np.std(X_train, axis=0)])
    
    # 正規化並訓練
    X_scaled = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-6)
    
    ae = build_autoencoder()
    ae.fit(X_scaled, X_scaled, epochs=50, batch_size=16, verbose=1)
    ae.save('home_model.keras')
    print("模型訓練完成並儲存為 home_model.keras")
