import torch
import torch.nn as nn
import torch.nn.functional as F

# --- PHẢI GIỮ LẠI CÁC CLASS ĐỂ LOAD MODEL ---
class ConvBlock(nn.Module):
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

class AttackDetector:
    LABELS = {0: "Benign (An toàn)", 1: "XSS Attack", 2: "SQL Injection"}
    
    def __init__(self, model_path='attack_model_best.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 512
        self.model = AttackNet(num_classes=3).to(self.device)
        
        # Load trọng số đã train
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        self.model.eval() # Chuyển sang chế độ dự đoán
        print(f"Đã load model từ {model_path} trên {self.device}")

    def predict(self, text):
        # Tiền xử lý text giống hệt lúc train
        tokens = [ord(c) if ord(c) < 256 else 1 for c in str(text)[:self.max_len]]
        tokens += [0] * (self.max_len - len(tokens))
        
        x = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            out = self.model(x)
            probs = F.softmax(out, dim=1)[0]
            pred = out.argmax(1).item()
            
        return {
            'text': text,
            'label': self.LABELS[pred],
            'confidence': f"{probs[pred].item()*100:.2f}%"
        }

# --- SỬ DỤNG ---
if __name__ == "__main__":
    detector = AttackDetector('attack_model_best.pt')

    # Danh sách thử nghiệm
    test_queries = [
        "SELECT * FROM users WHERE id = '1' OR '1'='1'",
        "<script>alert('XSS')</script>",
        "index.php?user=admin&session=123",
        "DROP TABLE students;--",
        "Hello, how are you today?"
    ]

    print("\n--- KẾT QUẢ KIỂM TRA ---")
    for q in test_queries:
        res = detector.predict(q)
        print(f"Input: {res['text']}\nResult: {res['label']} ({res['confidence']})\n{'-'*30}")
