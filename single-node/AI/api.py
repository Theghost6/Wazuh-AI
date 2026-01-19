import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ==============================================================================
# PHẦN 1: COPY KIẾN TRÚC MODEL TỪ FILE TRAIN
# (Bắt buộc phải có cả ConvBlock và AttackNet)
# ==============================================================================

class ConvBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.bn = nn.BatchNorm1d(out_ch)
    
    def forward(self, x): return F.relu(self.bn(self.conv(x)))

class AttackNet(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=128, num_filters=128, num_classes=3, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        kernels = [2, 3, 4, 5, 7]
        self.convs1 = nn.ModuleList([ConvBlock(embed_dim, num_filters, k) for k in kernels])
        self.convs2 = nn.ModuleList([ConvBlock(num_filters, num_filters, k) for k in kernels])
        self.pool = nn.AdaptiveMaxPool1d(1)
        total = num_filters * len(kernels)
        self.classifier = nn.Sequential(
            nn.Linear(total, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        outs = [self.pool(c2(c1(x))).squeeze(-1) for c1, c2 in zip(self.convs1, self.convs2)]
        return self.classifier(torch.cat(outs, dim=1))

# ==============================================================================
# PHẦN 2: CẤU HÌNH & XỬ LÝ DỮ LIỆU
# ==============================================================================

LABELS = {0: "Benign", 1: "XSS", 2: "SQLi"}
MAX_LEN = 512 # Mặc định theo code train

def text_to_tensor(text, max_len=512):
    """
    Chuyển text thành vector số (ASCII) giống hệt hàm trong AttackDataset
    """
    # Cắt chuỗi nếu quá dài
    text_str = str(text)[:max_len]
    
    # Convert sang ASCII code (0-255)
    tokens = [ord(c) if ord(c) < 256 else 1 for c in text_str]
    
    # Padding (điền số 0 vào cho đủ độ dài)
    tokens += [0] * (max_len - len(tokens))
    
    # Chuyển thành Tensor và thêm batch dimension [1, 512]
    return torch.tensor([tokens], dtype=torch.long)

# Class định nghĩa dữ liệu đầu vào cho API
class LogInput(BaseModel):
    log_content: str  # API sẽ nhận một chuỗi text (ví dụ: log url, user-agent...)

# ==============================================================================
# PHẦN 3: KHỞI TẠO SERVER & LOAD MODEL
# ==============================================================================
app = FastAPI()
model = None
device = torch.device("cpu") # Chạy API dùng CPU cho nhẹ

# Đường dẫn file model
MODEL_PATH = "./attack_model_best.pt"

print("--- ĐANG KHỞI ĐỘNG AI SERVER ---")
try:
    # 1. Khởi tạo kiến trúc
    model = AttackNet(num_classes=3)
    
    # 2. Load file .pt
    # Lưu ý: Code train save dict gồm {'model': ..., 'max_len': ...}
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # 3. Lấy state_dict từ key 'model'
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        # Cập nhật max_len nếu có trong file save
        if 'max_len' in checkpoint:
            MAX_LEN = checkpoint['max_len']
            print(f"Đã cập nhật Max Len từ model: {MAX_LEN}")
    else:
        # Fallback nếu save kiểu cũ
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval() # Bắt buộc: Tắt dropout/batchnorm mode
    print("✅ LOAD MODEL THÀNH CÔNG!")

except Exception as e:
    print(f"❌ LỖI LOAD MODEL: {e}")

# ==============================================================================
# PHẦN 4: API ENDPOINTS
# ==============================================================================

@app.get("/")
def home():
    return {"status": "AI Security Model Online", "labels": LABELS}

@app.post("/predict")
async def predict(input_data: LogInput):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        raw_text = input_data.log_content
        
        # 1. Preprocess
        tensor_input = text_to_tensor(raw_text, MAX_LEN).to(device)
        
        # 2. Predict
        with torch.no_grad():
            output = model(tensor_input)
            
            # Tính xác suất (Softmax)
            probs = F.softmax(output, dim=1)[0]
            
            # Lấy class có điểm cao nhất
            pred_index = output.argmax(1).item()
            pred_label = LABELS[pred_index]
            confidence = probs[pred_index].item()

        # 3. Trả kết quả chi tiết
        return {
            "prediction": pred_label,          # Kết quả: Benign/XSS/SQLi
            "is_attack": pred_index != 0,      # True nếu là XSS hoặc SQLi
            "confidence": round(confidence, 4),# Độ tin cậy (0.0 - 1.0)
            "details": {                       # Chi tiết % từng loại
                "Benign": round(probs[0].item(), 4),
                "XSS": round(probs[1].item(), 4),
                "SQLi": round(probs[2].item(), 4)
            }
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
