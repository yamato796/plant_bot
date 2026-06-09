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
def generate_expanded_drone(error_score, duration=0.1, sr=44100):
    """擴展響應範圍之音訊合成引擎"""
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # 1. 指數頻率映射 (50Hz ~ 1200Hz)
    min_f, max_f = 50.0, 1200.0
    base_freq = min_f * ((max_f / min_f) ** error_score)
    
    # 2. 基礎波與動態失真
    wave1 = np.sin(2 * np.pi * base_freq * t)
    detune_freq = base_freq * (1 + (error_score * 0.08)) 
    wave2 = np.sin(2 * np.pi * detune_freq * t) * error_score
    
    # 3. 高頻泛音注入 (當 error_score > 0.5 時激發，頻率為基頻 3 倍)
    harmonic_amp = np.clip((error_score - 0.5) * 2.0, 0.0, 1.0)
    wave3 = np.sin(2 * np.pi * (base_freq * 3.0) * t) * harmonic_amp
    
    # 4. LFO 脈衝調變 (速率隨分數提升：1Hz ~ 16Hz)
    lfo_rate = 1.0 + (error_score * 15.0)
    lfo = (np.sin(2 * np.pi * lfo_rate * t) + 1.0) / 2.0
    
    # 5. 陣列疊加與峰值正規化
    mixed = (wave1 + wave2 + wave3) * lfo
    
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
