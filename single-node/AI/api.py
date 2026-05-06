"""Wazuh-AI Attack Detection API v12 - Final Version"""  
"""Đầy đủ extraction + filtering để tránh false positive"""  
from fastapi import FastAPI  
from fastapi.responses import JSONResponse  
from pydantic import BaseModel  
import uvicorn  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import time  
import re  
import urllib.parse  
import os  

MODEL_PATH = "attack_model.pt"  
PORT = 5000  

# Patterns to skip (High-entropy session junk)
SKIP_PATTERNS = [
    "rudderencrypt:",
    "eyj", # JWT starting with 'eyJ'
    "u2fsdgvkx1", # Salted Base64 (OpenSSL)
    "wp-settings-",
    "rl_page_",
    "rl_anonymous_",
    "rl_user_",
    "n8n-auth",
    "stoken=",
    "csrf_token=",
    "sessionid="
]

# Keys to ignore (UI Noise)
IGNORE_KEYS = ["submit", "btnclear", "btnsubmit", "login", "button"]



class SEBlock(nn.Module):  
    def __init__(self, ch, r=16):  
        super().__init__()  
        self.fc = nn.Sequential(nn.Linear(ch, ch//r, bias=False), nn.ReLU(), nn.Linear(ch//r, ch, bias=False), nn.Sigmoid())  
        self.pool = nn.AdaptiveAvgPool1d(1)  
    def forward(self, x):  
        return x * self.fc(self.pool(x).view(x.size(0), -1)).view(x.size(0), -1, 1).expand_as(x)  

class ResBlock(nn.Module):  
    def __init__(self, in_ch, out_ch, k):  
        super().__init__()  
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding='same')  
        self.bn1 = nn.BatchNorm1d(out_ch)  
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding='same')  
        self.bn2 = nn.BatchNorm1d(out_ch)  
        self.se = SEBlock(out_ch)  
        self.shortcut = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch)) if in_ch != out_ch else nn.Sequential()  
    def forward(self, x):  
        return F.relu(self.shortcut(x) + self.se(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))  

class EnhancedCNN(nn.Module):  
    def __init__(self, vocab=256, emb=128, filters=256, classes=4):  
        super().__init__()  
        self.embed = nn.Embedding(vocab, emb, padding_idx=0)  
        self.drop = nn.Dropout(0.2)  
        self.branches = nn.ModuleList([  
            nn.Sequential(ResBlock(emb, filters, k), ResBlock(filters, filters, k), nn.AdaptiveMaxPool1d(1))  
            for k in [2, 3, 5, 7, 11]  
        ])  
        self.fc = nn.Sequential(  
            nn.Dropout(0.5), nn.Linear(filters * 5, 1024), nn.BatchNorm1d(1024), nn.GELU(),  
            nn.Dropout(0.3), nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, classes)  
        )  
    def forward(self, x):  
        x = self.drop(self.embed(x).transpose(1, 2))  
        return self.fc(torch.cat([b(x).squeeze(-1) for b in self.branches], 1))  



MAX_CANDIDATES = 12  

def select_candidates(payloads, log_line, limit=MAX_CANDIDATES):  
    """Reduce number of candidates (model-only)"""  
    seen = set()  
    candidates = []  
    for p in payloads:  
        if not p or len(p) <= 1:  
            continue  
        if p in seen:  
            continue  
        seen.add(p)  
        candidates.append(p)  
        if len(candidates) >= limit:  
            break  
    if log_line and log_line not in seen and len(candidates) < limit:  
        candidates.append(log_line)  
    return candidates  



def extract_payloads(log_line):  
    """Extract payloads: URL and Body only (Aggressive filtering)"""  
    payloads = []  
    decode = lambda s: urllib.parse.unquote_plus(s) if s else s  

    # 1. Extract params from URL Query String
    m = re.search(r'\?([^\"]+?)(?:\s+HTTP|\s*")', log_line)  
    if m:  
        for p in m.group(1).split('&'):  
            if '=' in p:  
                v = decode(p.split('=', 1)[1])  
                if v and len(v) > 1:  
                    payloads.append(v)  
                    v2 = decode(v)  
                    if v2 != v:  
                        payloads.append(v2)  

    # 2. Extract Body (last quoted field candidate)
    # We find all quoted strings
    quoted = re.findall(r'"([^"]*)"', log_line)
    if not quoted:
        return payloads

    # The request line is usually first, we analyze it for URL path
    req_line = quoted[0]
    # (Optional: we could extract path here, but detect_log uses the whole line anyway)

    # We look at remaining quoted fields for the POST body
    for field in reversed(quoted[1:]):
        token = (field or "").strip()
        low = token.lower()
        
        # Skip empty or dash
        if not token or token == "-": continue
        
        # AGGRESSIVE FILTER: Skip anything that looks like a header or session junk
        if any(low.startswith(h) for h in ["cookie:", "user-agent:", "referer:", "host:", "authorization:", "mozilla/", "http://", "https:"]):
            continue
        
        # SKIP patterns (JWT, Encryption, Tracking)
        if any(p in low for p in SKIP_PATTERNS):
            continue

        # If it contains '=' it is likely a form-data body
        if "=" in token:
            for p in token.split('&'):
                if '=' in p:
                    key, val = p.split('=', 1)
                    # Skip noise keys
                    if key.lower().strip() in IGNORE_KEYS:
                        continue
                    v = decode(val)
                    if v and len(v) > 1:
                        payloads.append(v)
        # If it looks like JSON
        elif token.startswith('{'):
            try:
                import json
                data = json.loads(token.replace('\\"', '"'))
                def get_vals(obj):
                    if isinstance(obj, dict):
                        return [v for val in obj.values() for v in get_vals(val)]
                    elif isinstance(obj, list):
                        return [v for item in obj for v in get_vals(item)]
                    elif isinstance(obj, str) and len(obj) > 1:
                        return [obj]
                    return []
                for v in get_vals(data):
                    payloads.append(v)
            except:
                payloads.append(token)
        
    return [p for p in payloads if p and len(p) > 1]  



class AttackDetector:  
    LABELS = {0: "Benign", 1: "XSS", 2: "SQLi", 3: "CMDi"}
    
    def __init__(self, model_path=MODEL_PATH, max_len=512):  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.max_len = max_len  
        self.model = EnhancedCNN(classes=4).to(self.device)  
        
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)  
        if os.path.exists(path):  
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))  
            self.model.eval()  
            print(f"Model loaded: {path} ({self.device})")  


    def tokenize(self, text):  
        tokens = [ord(c) if ord(c) < 256 else 1 for c in str(text)[:self.max_len]]  
        return torch.tensor([tokens + [0] * (self.max_len - len(tokens))], dtype=torch.long).to(self.device)  

    def predict_single(self, text: str) -> dict:  
        """Predict single text"""  
        with torch.no_grad():  
            probs = F.softmax(self.model(self.tokenize(text)), dim=1)[0]  
            pred = probs.argmax().item()  
        
        label = self.LABELS[pred]
        return {  
            "label": label,  
            "is_attack": pred != 0,  
            "attack_type": label if pred != 0 else None,  
            "confidence": round(probs[pred].item() * 100, 2),  
            "detected_payload": text[:100],  
            "probabilities": {l: round(probs[i].item() * 100, 2) for i, l in self.LABELS.items()}  
        }  

    def detect_log(self, log_line: str, threshold=0.5, provided_candidates=None) -> dict:  
        """Detect attack trong HTTP log"""  
        best = {"pred": 0, "conf": 0, "payload": None, "probs": None}  

        if provided_candidates:
            candidates = select_candidates(provided_candidates, log_line)
        else:
            candidates = select_candidates(extract_payloads(log_line), log_line)  
        for payload in candidates:  
            with torch.no_grad():  
                probs = F.softmax(self.model(self.tokenize(payload)), dim=1)[0]  
                pred = probs.argmax().item()  
            
            if pred > 0 and probs[pred].item() > best["conf"] and probs[pred].item() > threshold:  
                best = {"pred": pred, "conf": probs[pred].item(), "payload": payload, "probs": probs}  
        
        if best["pred"] > 0:  
            return {  
                "label": self.LABELS[best["pred"]], "is_attack": True,  
                "attack_type": self.LABELS[best["pred"]],  
                "confidence": round(best["conf"] * 100, 2),  
                "detected_payload": best["payload"][:100] if best["payload"] else None,  
                "probabilities": {l: round(best["probs"][i].item() * 100, 2) for i, l in self.LABELS.items()}  
            }  
        return {"label": "Benign", "is_attack": False, "attack_type": None, "confidence": 100.0,  
                "detected_payload": None, "probabilities": {"Benign": 100.0, "XSS": 0.0, "SQLi": 0.0, "CMDi": 0.0}}  



app = FastAPI(title="Wazuh-AI v12")  
detector = AttackDetector()  

class Request(BaseModel):  
    text: str = None  
    log: str = None  
    payloads: list = None
    threshold: float = 0.5  

@app.get("/")  
def home():  
    return {"status": "active", "model": "v12_final"}  

@app.post("/predict")  
def predict(req: Request):  
    """Predict - auto-detect if input is log or raw payload"""  
    start = time.time()  
    
    # Priority 1: Use pre-extracted payloads if provided (from custom-ai)
    if req.payloads is not None:
        res = detector.detect_log(req.log or "", req.threshold, provided_candidates=req.payloads)
    
    # Priority 2: Use raw log if provided
    elif req.log:
        res = detector.detect_log(req.log, req.threshold)
    
    # Priority 3: Use text as raw payload
    elif req.text:
        res = detector.predict_single(req.text)
    
    else:
        return JSONResponse(status_code=400, content={"error": "Missing log/payloads/text"})  
    
    if res["is_attack"]:  
        print(f"[ATTACK] {res['attack_type']} ({res['confidence']}%)")  
    return {**res, "time_ms": round((time.time() - start) * 1000, 2)}  

@app.post("/detect")  
def detect(req: Request):  
    """Detect attack in HTTP log"""  
    start = time.time()  
    if not req.log:  
        return JSONResponse(status_code=400, content={"error": "Missing log"})  
    res = detector.detect_log(req.log, req.threshold)  
    if res["is_attack"]:  
        print(f"[ATTACK] {res['attack_type']} ({res['confidence']}%) - {res.get('detected_payload', '')[:50]}")  
    return {**res, "time_ms": round((time.time() - start) * 1000, 2)}  

if __name__ == "__main__":  
    print(f"Starting on port {PORT}...")  
    uvicorn.run(app, host="0.0.0.0", port=PORT)  

