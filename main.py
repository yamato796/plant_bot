import subprocess
import re
import time
import hashlib
import threading
import numpy as np
import sounddevice as sd

# --- 系統常數設定 ---
SAMPLE_RATE = 44100
BUFFER_DURATION = 0.1
SCAN_INTERVAL = 2.0  # 環境掃描頻率 (秒)

# 共享狀態與執行緒鎖
active_nodes = {}
state_lock = threading.Lock()

# --- 模組：環境掃描與解析 ---
def scan_environment():
    while True:
        try:
            # 呼叫底層網卡指令掃描 AP
            cmd = "sudo iw dev wlan0 scan"
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')
            
            macs = re.findall(r'BSS ([0-9a-fA-F:]+)\(on', result)
            signals = re.findall(r'signal: (-\d+\.\d+) dBm', result)
            
            current_time = time.time()
            
            with state_lock:
                for mac, sig in zip(macs, signals):
                    rssi = float(sig)
                    
                    # 頻率計算：MAC 雜湊值映射至 100Hz ~ 800Hz
                    hash_val = int(hashlib.md5(mac.encode()).hexdigest(), 16)
                    freq = 100 + (hash_val % 700)
                    
                    # 振幅計算：訊號強度映射至 0.0 ~ 1.0
                    amp = np.clip((rssi - (-90)) / 60.0, 0.0, 1.0)
                    
                    # 相位計算：取 MAC 尾碼映射為左右聲道比例 (0.0=極左, 1.0=極右)
                    last_byte_hex = mac.split(':')[-1]
                    pan = int(last_byte_hex, 16) / 255.0 
                    
                    active_nodes[mac] = {
                        'freq': freq, 
                        'amp': amp, 
                        'pan': pan, 
                        'last_seen': current_time
                    }
        except Exception:
            pass # 略過執行期錯誤，確保背景迴圈不中斷
            
        time.sleep(SCAN_INTERVAL)

# --- 模組：音訊合成與相位平移 ---
def generate_audio(duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    mixed_l = np.zeros_like(t)
    mixed_r = np.zeros_like(t)
    
    with state_lock:
        current_time = time.time()
        stale_macs = []
        
        for mac, data in active_nodes.items():
            # 逾時節點清理機制
            if current_time - data['last_seen'] > 5.0:
                stale_macs.append(mac)
                continue
            
            base_wave = np.sin(2 * np.pi * data['freq'] * t) * data['amp']
            
            # 依 Pan 值分配雙聲道能量
            mixed_l += base_wave * (1.0 - data['pan'])
            mixed_r += base_wave * data['pan']
            
        for mac in stale_macs:
            del active_nodes[mac]

    # 輸出前防破音處理 (Peak Normalization)
    max_val = max(np.max(np.abs(mixed_l)), np.max(np.abs(mixed_r)))
    if max_val > 0:
        mixed_l = mixed_l / max_val
        mixed_r = mixed_r / max_val
        
    return np.float32(np.column_stack((mixed_l, mixed_r)))

# --- 系統主程式 ---
def main():
    print("系統啟動：開始環境 Wi-Fi 頻譜掃描與雙聲道音訊映射...")
    
    # 啟動背景掃描執行緒
    scanner_thread = threading.Thread(target=scan_environment, daemon=True)
    scanner_thread.start()
    
    try:
        while True:
            # 阻擋式播放迴圈
            audio_buffer = generate_audio(BUFFER_DURATION)
            sd.play(audio_buffer, SAMPLE_RATE)
            sd.wait()
            
    except KeyboardInterrupt:
        print("\n接收到終止訊號，系統關閉。")
        sd.stop()

if __name__ == "__main__":
    main()
