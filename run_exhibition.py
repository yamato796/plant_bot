import argparse
import time
import pickle
import numpy as np
import sounddevice as sd
from sklearn.preprocessing import StandardScaler
from train_model import get_wifi_features

# --- 參數解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='home_model.pkl')
args = parser.parse_args()

# --- 音訊合成引擎 ---
def generate_drone(error_score, duration=0.1, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), False)
    
    base_freq = 100 + (error_score * 500)
    wave1 = np.sin(2 * np.pi * base_freq * t)
    
    detune_freq = base_freq * (1 + (error_score * 0.05))
    wave2 = np.sin(2 * np.pi * detune_freq * t) * error_score
    
    mixed = wave1 + wave2
    
    # 峰值正規化強制推動最大硬體增益
    max_amp = np.max(np.abs(mixed))
    if max_amp > 0: 
        mixed /= max_amp
        
    return np.float32(np.column_stack((mixed, mixed)))

if __name__ == "__main__":
    # 強制指定音訊介面 (請確認 DAC ID 為 1)
    sd.default.device = 1
    
    # 載入預訓練模型
    with open(args.model, 'rb') as f:
        payload = pickle.load(f)
    model = payload['model']

    # --- 展場基線校正 (Domain Adaptation) ---
    print("啟動展場基準線靜默校正 (預估耗時 600 秒)...")
    ex_data = []
    for i in range(300):
        ex_data.append(get_wifi_features())
        time.sleep(2)
        if (i + 1) % 30 == 0:
            print(f"校正進度: {i + 1} / 300")
            
    # 實體化展場專用特徵轉換器
    ex_scaler = StandardScaler()
    ex_scaler.fit(ex_data)

    print("校正完畢，進入實時推論迴圈。")
    
    try:
        while True:
            # 1. 資料擷取與展場特徵正規化
            live_vector = np.array(get_wifi_features()).reshape(1, -1)
            live_scaled = ex_scaler.transform(live_vector)
            
            # 2. 異常分數推論 (負值，越小代表越異常)
            score = model.score_samples(live_scaled)[0]
            
            # 3. 分數正規化裁剪 (-0.4 為基準正常值，隨環境波動向負數遞減)
            error_score = np.clip((-score - 0.4) / 0.4, 0.0, 1.0) 
            
            # 4. 合成音訊並阻塞播放
            audio_buffer = generate_drone(error_score)
            sd.play(audio_buffer, 44100)
            sd.wait()
            
    except KeyboardInterrupt:
        print("\n中斷執行緒。")
        sd.stop()
