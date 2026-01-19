#!/usr/bin/env python3
import sys
import json
import requests
import time

# 1. Đọc dữ liệu alert từ Wazuh
try:
    alert_file = sys.argv[1]
    user = sys.argv[2]
    passwd = sys.argv[3]
    with open(alert_file) as f:
        alert_json = json.load(f)
except Exception as e:
    sys.exit(1)

# 2. Lấy nội dung log cần soi (full_log hoặc data.url, data.command...)
# Tùy rule mà log nằm ở full_log hay data
log_content = ""
if 'full_log' in alert_json:
    log_content = alert_json['full_log']
elif 'data' in alert_json and 'log' in alert_json['data']:
    log_content = alert_json['data']['log']
else:
    # Lấy đại một trường nào đó nếu không tìm thấy log chuẩn
    log_content = str(alert_json)

# 3. Gửi sang AI API
# THAY 172.17.0.1 BẰNG IP BẠN TÌM ĐƯỢC Ở BƯỚC 1
AI_API_URL = "http://172.17.0.1:5000/predict" 

payload = {"log_content": log_content}
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(AI_API_URL, json=payload, headers=headers, timeout=2)
    
    if response.status_code == 200:
        result = response.json()
        
        # 4. Nếu AI phát hiện tấn công (is_attack = True)
        if result.get("is_attack"):
            # Tạo log mới để Wazuh ghi nhận lại kết quả từ AI
            ai_alert = {
                "timestamp": alert_json.get("timestamp"),
                "ai_prediction": result["prediction"],
                "ai_confidence": result["confidence"],
                "original_log": log_content,
                "src_ip": alert_json.get("data", {}).get("srcip", "unknown"),
                "description": f"AI DETECTED ATTACK: {result['prediction']}"
            }
            
            # Ghi vào file active-responses.log (Wazuh tự động đọc file này)
            with open('/var/ossec/logs/active-responses.log', 'a') as log_file:
                log_file.write(json.dumps(ai_alert) + "\n")
                
except Exception as e:
    # Ghi log lỗi nếu cần debug
    with open('/var/ossec/logs/active-responses.log', 'a') as log_file:
        log_file.write(json.dumps({"error": str(e)}) + "\n")
