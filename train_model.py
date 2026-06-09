import time
import subprocess
import re
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def get_wifi_features():
    """擷取環境特徵矩陣 (AP總數, 平均訊號強度, 訊號標準差)"""
    try:
        cmd = "sudo iw dev wlan0 scan"
        res = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')
        signals = [float(s) for s in re.findall(r'signal: (-\d+\.\d+) dBm', res)]
        if not signals: 
            return [0.0, -90.0, 0.0]
        return [len(signals), np.mean(signals), np.std(signals)]
    except Exception:
        return [0.0, -90.0, 0.0]

if __name__ == "__main__":
    print("啟動訓練：收集背景特徵 (預計 10 分鐘)...")
    data = []
    
    # 執行 300 次採樣，每次間隔 2 秒
    for i in range(300):
        data.append(get_wifi_features())
        time.sleep(2)
        if (i + 1) % 30 == 0:
            print(f"進度: {i + 1} / 300")

    X_train = np.array(data)

    # 特徵標準化 (Z-Score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 訓練無監督異常偵測模型 (設定污染率 5%)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)

    # 序列化封裝模型與預設的 Scaler
    with open('home_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
        
    print("模型封裝完畢並儲存為 home_model.pkl")
