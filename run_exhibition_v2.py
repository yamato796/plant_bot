import argparse
import time
import pickle
import subprocess
import re
import hashlib
import numpy as np
import sounddevice as sd
from sklearn.preprocessing import StandardScaler

# --- 參數解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='home_model.pkl')
args = parser.parse_args()

# --- 模組：環境實時掃描 ---
def scan_live_environment():
    """擷取環境特徵矩陣與原始 AP 列表"""
    try:
        cmd = "sudo iw dev wlan0 scan | grep -E 'signal:|BSS'"
        res = subprocess.check_output(cmd, shell=True, timeout=10).decode('utf-8')
        
        macs = re.findall(r'BSS ([0-9a-fA-F:]+)\(on', res)
        signals = [float(s) for s in re.findall(r'signal: (-\d+\.\d+)', res)]
        
        ap_data = [{'mac': m, 'rssi': s} for m, s in zip(macs, signals)]
        
        if not signals: 
            return [0.0, -90.0, 0.0], []
            
        features = [len(signals), np.mean(signals), np.std(signals)]
        return features, ap_data
        
    except Exception:
        return [0.0, -90.0, 0.0], []

# --- 模組：混合音訊合成引擎 ---
def generate_hybrid_drone(ap_data, error_score, duration=0.1, sr=44100):
    """融合 Rule-based 加法合成與 ML 異常分數調變"""
    t = np.linspace(0, duration, int(sr * duration), False)
    mixed = np.zeros_like(t)
    
    for ap in ap_data:
        # 1. 空間固有頻率：MAC 雜湊決定基頻 (上移至 200Hz-1000Hz 區間)
        hash_val = int(hashlib.md5(ap['mac'].encode()).hexdigest(), 16)
        base_freq = 200 + (hash_val % 800)
        
        # 2. 基礎能量：RSSI 決定基礎振幅
        amp = np.clip((ap['rssi'] - (-90)) / 60.0, 0.0, 1.0)
        
        # 3. 異常干擾：連動頻率偏移與高頻失真
        shift_freq = base_freq * (1.0 + (error_score * 0.5))
        detune_freq = shift_freq * (1.0 + (error_score * 0.15))
        
        wave1 = np.sin(2 * np.pi * shift_freq * t) * amp
        wave2 = np.sin(2 * np.pi * detune_freq * t) * (amp * error_score)
        
        mixed += (wave1 + wave2)
        
    # 4. 強制峰值正規化：確保 100% 推動單體
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
    print("啟動展場基準線靜默校正...")
    ex_data = []
    
    # 沿用原有的 5 次快速校正進行測試
    for i in range(5):
        features, _ = scan_live_environment()
        ex_data.append(features)
        time.sleep(2)
        if (i + 1) % 1 == 0:
            print(f"校正進度: {i + 1} / 5")
            
    # 實體化展場專用特徵轉換器
    ex_scaler = StandardScaler()
    ex_scaler.fit(ex_data)

    print("校正完畢，進入實時推論迴圈。")
    
    try:
        while True:
            # 1. 擷取實時特徵與原始 AP 陣列
            live_vector_data, current_ap_data = scan_live_environment()
            live_vector = np.array(live_vector_data).reshape(1, -1)
            
            # 2. 展場特徵正規化與推論
            live_scaled = ex_scaler.transform(live_vector)
            score = model.score_samples(live_scaled)[0]
            
            # 3. 分數裁剪映射
            error_score = np.clip((-score - 0.4) / 0.4, 0.0, 1.0) 
            
            # 4. 傳入陣列與分數合成音訊並阻塞播放
            audio_buffer = generate_hybrid_drone(current_ap_data, error_score)
            sd.play(audio_buffer, 44100)
            sd.wait()
            
    except KeyboardInterrupt:
        print("\n中斷執行緒。")
        sd.stop()
