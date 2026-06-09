import argparse
import time
import numpy as np
import tensorflow as tf
import sounddevice as sd
from train_model import get_wifi_features # 共用擷取邏輯

# --- 參數解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='home_model.keras', help='指定模型路徑')
args = parser.parse_args()

# --- 展場動態校正 ---
def calibrate_exhibition(duration=600):
    print(f"啟動展場基準線校正 ({duration} 秒)... 請保持空間淨空。")
    exhibition_data = []
    start = time.time()
    while time.time() - start < duration:
        exhibition_data.append(get_wifi_features())
        time.sleep(2)
    
    ex_mean = np.mean(exhibition_data, axis=0)
    ex_std = np.std(exhibition_data, axis=0) + 1e-6
    print("展場基準線建立完成。")
    return ex_mean, ex_std

# --- 音訊生成邏輯 ---
def generate_drone(error_score, duration=0.1, sr=44100):
    """依據重建誤差生成聲音"""
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # 基礎頻率與誤差成正比 (例如 100Hz 到 600Hz)
    base_freq = 100 + (error_score * 500)
    wave1 = np.sin(2 * np.pi * base_freq * t)
    
    # 誤差越大，加入越多失真與拍頻 (Beating)
    detune_freq = base_freq * (1 + (error_score * 0.05))
    wave2 = np.sin(2 * np.pi * detune_freq * t) * error_score
    
    mixed = (wave1 + wave2) / 2.0
    return np.float32(np.column_stack((mixed, mixed)))

if __name__ == "__main__":
    # 載入模型
    model = tf.keras.models.load_model(args.model)
    # 執行展場校正
    ex_mean, ex_std = calibrate_exhibition(600)
    
    print("系統進入即時推論模式。")
    try:
        while True:
            live_features = get_wifi_features()
            # 使用展場基準正規化，套用家中模型的推論邏輯
            x_live = (live_features - ex_mean) / ex_std
            x_live = np.expand_dims(x_live, axis=0)
            
            # 計算重建誤差 (Anomaly Score)
            reconstruction = model.predict(x_live, verbose=0)
            mse = np.mean(np.square(x_live - reconstruction))
            
            # 將誤差截斷並映射至 0.0 ~ 1.0 作為聲音參數
            error_score = np.clip(mse / 5.0, 0.0, 1.0) 
            
            audio = generate_drone(error_score)
            sd.play(audio, 44100)
            sd.wait()
            
    except KeyboardInterrupt:
        sd.stop()
