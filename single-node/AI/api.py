"""Wazuh-AI Attack Detection API v12 - Final Version"""  # Mô tả nhanh về module API.
"""Đầy đủ extraction + filtering để tránh false positive"""  # Ghi chú mục tiêu xử lý payload.
from fastapi import FastAPI  # Import FastAPI để tạo REST API.
from fastapi.responses import JSONResponse  # Import JSONResponse để trả lỗi JSON tùy chỉnh.
from pydantic import BaseModel  # Import BaseModel để định nghĩa schema request.
import uvicorn  # Import uvicorn để chạy server ASGI.
import torch  # Import PyTorch cho model ML.
import torch.nn as nn  # Import nn để dùng các lớp neural network.
import torch.nn.functional as F  # Import functional để dùng hàm activation/softmax.
import time  # Import time để đo thời gian xử lý.
import re  # Import regex để trích xuất payload từ log.
import urllib.parse  # Import parser để decode URL.
import os  # Import os để xử lý đường dẫn file.

MODEL_PATH = "attack_model.pt"  # Đường dẫn file trọng số model.
PORT = 5000  # Cổng chạy API.


# ============ MODEL ============  # Phần định nghĩa mô hình.
class SEBlock(nn.Module):  # Khối Squeeze-and-Excitation.
    def __init__(self, ch, r=16):  # Hàm khởi tạo với số kênh và tỷ lệ nén.
        super().__init__()  # Gọi khởi tạo lớp cha.
        self.fc = nn.Sequential(nn.Linear(ch, ch//r, bias=False), nn.ReLU(), nn.Linear(ch//r, ch, bias=False), nn.Sigmoid())  # MLP để tạo trọng số kênh.
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool trung bình để gom thông tin theo kênh.
    def forward(self, x):  # Hàm forward của SEBlock.
        return x * self.fc(self.pool(x).view(x.size(0), -1)).view(x.size(0), -1, 1).expand_as(x)  # Scale kênh rồi nhân với input.

class ResBlock(nn.Module):  # Khối residual 1D.
    def __init__(self, in_ch, out_ch, k):  # Khởi tạo với kênh vào/ra và kernel size.
        super().__init__()  # Gọi khởi tạo lớp cha.
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding='same')  # Conv1D đầu tiên.
        self.bn1 = nn.BatchNorm1d(out_ch)  # BatchNorm cho conv1.
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding='same')  # Conv1D thứ hai.
        self.bn2 = nn.BatchNorm1d(out_ch)  # BatchNorm cho conv2.
        self.se = SEBlock(out_ch)  # Khối SE để recalibrate kênh.
        self.shortcut = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch)) if in_ch != out_ch else nn.Sequential()  # Nhánh tắt khi đổi kênh.
    def forward(self, x):  # Hàm forward của ResBlock.
        return F.relu(self.shortcut(x) + self.se(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))  # Cộng residual rồi activation.

class EnhancedCNN(nn.Module):  # Mô hình CNN nhiều nhánh.
    def __init__(self, vocab=256, emb=128, filters=256, classes=4):  # Khởi tạo với vocab/embedding/filter/classes.
        super().__init__()  # Gọi khởi tạo lớp cha.
        self.embed = nn.Embedding(vocab, emb, padding_idx=0)  # Embedding theo ký tự.
        self.drop = nn.Dropout(0.2)  # Dropout tránh overfit.
        self.branches = nn.ModuleList([  # Tạo danh sách các nhánh conv.
            nn.Sequential(ResBlock(emb, filters, k), ResBlock(filters, filters, k), nn.AdaptiveMaxPool1d(1))  # 2 ResBlock + maxpool.
            for k in [2, 3, 5, 7, 11]  # Các kích thước kernel khác nhau.
        ])  # Kết thúc ModuleList.
        self.fc = nn.Sequential(  # MLP phân loại cuối.
            nn.Dropout(0.5), nn.Linear(filters * 5, 1024), nn.BatchNorm1d(1024), nn.GELU(),  # Tầng đầu MLP.
            nn.Dropout(0.3), nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, classes)  # Tầng cuối MLP.
        )  # Kết thúc MLP.
    def forward(self, x):  # Hàm forward của mô hình.
        x = self.drop(self.embed(x).transpose(1, 2))  # Embedding rồi đổi trục để conv.
        return self.fc(torch.cat([b(x).squeeze(-1) for b in self.branches], 1))  # Ghép nhánh rồi đưa vào FC.


# ============ HELPER FUNCTIONS ============  # Phần hàm hỗ trợ.
MAX_CANDIDATES = 12  # Số payload tối đa đưa vào model.

def select_candidates(payloads, log_line, limit=MAX_CANDIDATES):  # Lọc payloads để giảm số lượng.
    """Reduce number of candidates (model-only)"""  # Docstring mô tả ngắn.
    seen = set()  # Tập để tránh trùng payload.
    candidates = []  # Danh sách payload được chọn.
    for p in payloads:  # Duyệt từng payload.
        if not p or len(p) <= 1:  # Bỏ payload rỗng hoặc quá ngắn.
            continue  # Bỏ qua vòng lặp.
        if p in seen:  # Nếu đã gặp payload.
            continue  # Bỏ qua để tránh trùng.
        seen.add(p)  # Đánh dấu payload đã thấy.
        candidates.append(p)  # Thêm vào danh sách ứng viên.
        if len(candidates) >= limit:  # Nếu đủ số lượng.
            break  # Dừng sớm.
    if log_line and log_line not in seen and len(candidates) < limit:  # Nếu còn slot và log chưa có.
        candidates.append(log_line)  # Thêm nguyên log làm ứng viên.
    return candidates  # Trả về danh sách ứng viên.


# ============ PAYLOAD EXTRACTOR ============  # Phần trích payload.
def extract_payloads(log_line):  # Trích payload từ dòng log.
    """Extract payloads: URL, Referer, Body (gọn)"""  # Docstring mô tả nguồn trích.
    payloads = []  # Danh sách payload trích được.
    decode = lambda s: urllib.parse.unquote_plus(s) if s else s  # Hàm decode URL.

    # 1. URL query params  # Bước trích query string.
    m = re.search(r'\?([^\"]+?)(?:\s+HTTP|\s*")', log_line)  # Tìm phần query trong URL.
    if m:  # Nếu match.
        for p in m.group(1).split('&'):  # Tách từng cặp tham số.
            if '=' in p:  # Nếu đúng dạng key=value.
                v = decode(p.split('=', 1)[1])  # Lấy giá trị và decode.
                if v and len(v) > 1:  # Nếu giá trị hợp lệ.
                    payloads.append(v)  # Thêm payload.
                    v2 = decode(v)  # Decode lần nữa để xử lý double-encoding.
                    if v2 != v:  # Nếu khác.
                        payloads.append(v2)  # Thêm phiên bản decode lần 2.

    # 2. Form body in middle  # Bước trích body ở giữa log.
    m = re.search(r'"\s+([a-zA-Z_]+=[^\"]+)$', log_line)  # Tìm chuỗi dạng form.
    if m:  # Nếu match.
        for p in m.group(1).split('&'):  # Tách từng cặp.
            if '=' in p:  # Nếu đúng dạng key=value.
                payloads.append(decode(p.split('=', 1)[1]))  # Thêm giá trị decode.

    # 3. POST body at end  # Bước trích body cuối dòng.
    m = re.search(r'"([^\"]+)"\s*$', log_line)  # Lấy chuỗi cuối trong dấu nháy.
    if m:  # Nếu match.
        body = m.group(1)  # Lấy nội dung body.
        if body and body != '-' and not body.startswith('Mozilla') and not body.startswith('http'):  # Loại trừ body không hữu ích.
            if body.strip().startswith('{'):  # Nếu body là JSON.
                try:  # Thử parse JSON.
                    import json  # Import json trong scope nhỏ.
                    data = json.loads(body.replace('\\"', '"'))  # Parse JSON sau khi sửa escape.
                    def get_vals(obj):  # Hàm đệ quy lấy tất cả giá trị.
                        if isinstance(obj, dict):  # Nếu là dict.
                            return [v for val in obj.values() for v in get_vals(val)]  # Lấy giá trị con.
                        elif isinstance(obj, list):  # Nếu là list.
                            return [v for item in obj for v in get_vals(item)]  # Lấy giá trị từ list.
                        elif isinstance(obj, str) and len(obj) > 1:  # Nếu là chuỗi dài.
                            return [obj]  # Trả về chính chuỗi.
                        return []  # Trường hợp khác trả rỗng.
                    for v in get_vals(data):  # Duyệt mọi giá trị JSON.
                        payloads.append(v)  # Thêm vào payloads.
                except:  # Nếu parse lỗi.
                    payloads.append(body)  # Thêm nguyên body.
            elif '=' in body:  # Nếu body là dạng form.
                for p in body.split('&'):  # Tách từng cặp.
                    if '=' in p:  # Nếu đúng dạng key=value.
                        v = decode(p.split('=', 1)[1])  # Lấy giá trị decode.
                        if v and len(v) > 1:  # Nếu hợp lệ.
                            payloads.append(v)  # Thêm payload.
            elif len(body) > 2:  # Nếu body là chuỗi dài đơn.
                payloads.append(body)  # Thêm nguyên body.

    return [p for p in payloads if p and len(p) > 1]  # Trả payload hợp lệ.


# ============ DETECTOR ============  # Phần detector.
class AttackDetector:  # Lớp phát hiện tấn công.
    LABELS = {0: "Benign", 1: "XSS", 2: "SQLi", 3: "CMDi"}  # Mapping nhãn.
    
    def __init__(self, model_path=MODEL_PATH, max_len=512):  # Khởi tạo detector.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Chọn device GPU/CPU.
        self.max_len = max_len  # Độ dài tối đa chuỗi.
        self.model = EnhancedCNN(classes=4).to(self.device)  # Tạo model và đưa lên device.
        
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)  # Tính đường dẫn file model.
        if os.path.exists(path):  # Nếu file tồn tại.
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))  # Nạp trọng số.
            self.model.eval()  # Chuyển sang mode eval.
            print(f"Model loaded: {path} ({self.device})")  # Log thông tin model.

    def tokenize(self, text):  # Chuyển text thành tensor số.
        tokens = [ord(c) if ord(c) < 256 else 1 for c in str(text)[:self.max_len]]  # Map ký tự sang mã ASCII.
        return torch.tensor([tokens + [0] * (self.max_len - len(tokens))], dtype=torch.long).to(self.device)  # Pad và đưa lên device.

    def predict_single(self, text: str) -> dict:  # Dự đoán cho 1 chuỗi.
        """Predict single text"""  # Docstring mô tả.
        with torch.no_grad():  # Tắt gradient để suy luận nhanh.
            probs = F.softmax(self.model(self.tokenize(text)), dim=1)[0]  # Tính xác suất softmax.
            pred = probs.argmax().item()  # Lấy nhãn có xác suất cao nhất.
        return {  # Trả kết quả dạng dict.
            "label": self.LABELS[pred],  # Nhãn dự đoán.
            "is_attack": pred != 0,  # Có phải tấn công không.
            "attack_type": self.LABELS[pred] if pred != 0 else None,  # Loại tấn công.
            "confidence": round(probs[pred].item() * 100, 2),  # Độ tin cậy %.
            "detected_payload": text[:100],  # Trích payload ngắn.
            "probabilities": {l: round(probs[i].item() * 100, 2) for i, l in self.LABELS.items()}  # Xác suất từng lớp.
        }  # Kết thúc dict.

    def detect_log(self, log_line: str, threshold=0.5) -> dict:  # Phát hiện tấn công trong log.
        """Detect attack trong HTTP log"""  # Docstring mô tả.
        best = {"pred": 0, "conf": 0, "payload": None, "probs": None}  # Lưu kết quả tốt nhất.

        candidates = select_candidates(extract_payloads(log_line), log_line)  # Lấy danh sách payload ứng viên.
        for payload in candidates:  # Duyệt từng payload.
            with torch.no_grad():  # Tắt gradient.
                probs = F.softmax(self.model(self.tokenize(payload)), dim=1)[0]  # Tính xác suất.
                pred = probs.argmax().item()  # Lấy nhãn lớn nhất.
            
            if pred > 0 and probs[pred].item() > best["conf"] and probs[pred].item() > threshold:  # Điều kiện chọn tốt nhất.
                best = {"pred": pred, "conf": probs[pred].item(), "payload": payload, "probs": probs}  # Cập nhật best.
        
        if best["pred"] > 0:  # Nếu phát hiện tấn công.
            return {  # Trả kết quả tấn công.
                "label": self.LABELS[best["pred"]], "is_attack": True,  # Nhãn và cờ tấn công.
                "attack_type": self.LABELS[best["pred"]],  # Loại tấn công.
                "confidence": round(best["conf"] * 100, 2),  # Độ tin cậy %.
                "detected_payload": best["payload"][:100] if best["payload"] else None,  # Payload ngắn.
                "probabilities": {l: round(best["probs"][i].item() * 100, 2) for i, l in self.LABELS.items()}  # Xác suất từng lớp.
            }  # Kết thúc dict.
        return {"label": "Benign", "is_attack": False, "attack_type": None, "confidence": 100.0,  # Trả kết quả an toàn.
                "detected_payload": None, "probabilities": {"Benign": 100.0, "XSS": 0.0, "SQLi": 0.0, "CMDi": 0.0}}  # Xác suất mặc định.


# ============ API ============  # Phần API.
app = FastAPI(title="Wazuh-AI v12")  # Tạo app FastAPI.
detector = AttackDetector()  # Khởi tạo detector dùng chung.

class Request(BaseModel):  # Schema request.
    text: str = None  # Trường text input.
    log: str = None  # Trường log input.
    threshold: float = 0.5  # Ngưỡng confidence.

@app.get("/")  # Route GET root.
def home():  # Handler cho root.
    return {"status": "active", "model": "v12_final"}  # Trả trạng thái.

@app.post("/predict")  # Route POST predict.
def predict(req: Request):  # Handler dự đoán.
    """Predict - auto-detect if input is log or raw payload"""  # Docstring mô tả.
    start = time.time()  # Bắt đầu đo thời gian.
    content = req.text or req.log  # Lấy input ưu tiên text.
    if not content:  # Nếu thiếu input.
        return JSONResponse(status_code=400, content={"error": "Missing text/log"})  # Trả lỗi 400.
    
    # Auto-detect: if looks like HTTP log, use log detection  # Tự động chọn kiểu xử lý.
    if 'HTTP/' in content or (len(content) > 50 and re.search(r'\[\d+/\w+/\d+:', content)):  # Heuristic nhận diện log.
        res = detector.detect_log(content, req.threshold)  # Dùng detect_log.
    else:  # Nếu không phải log.
        res = detector.predict_single(content)  # Dùng predict_single.
    
    if res["is_attack"]:  # Nếu là tấn công.
        print(f"[ATTACK] {res['attack_type']} ({res['confidence']}%)")  # Log ra console.
    return {**res, "time_ms": round((time.time() - start) * 1000, 2)}  # Trả kết quả kèm thời gian.

@app.post("/detect")  # Route POST detect.
def detect(req: Request):  # Handler detect log.
    """Detect attack in HTTP log"""  # Docstring mô tả.
    start = time.time()  # Bắt đầu đo thời gian.
    if not req.log:  # Nếu thiếu log.
        return JSONResponse(status_code=400, content={"error": "Missing log"})  # Trả lỗi 400.
    res = detector.detect_log(req.log, req.threshold)  # Chạy detect.
    if res["is_attack"]:  # Nếu phát hiện tấn công.
        print(f"[ATTACK] {res['attack_type']} ({res['confidence']}%) - {res.get('detected_payload', '')[:50]}")  # Log payload ngắn.
    return {**res, "time_ms": round((time.time() - start) * 1000, 2)}  # Trả kết quả kèm thời gian.

if __name__ == "__main__":  # Điểm vào khi chạy trực tiếp.
    print(f"Starting on port {PORT}...")  # Log port.
    uvicorn.run(app, host="0.0.0.0", port=PORT)  # Chạy server.

