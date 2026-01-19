"""
Multi-Class Attack Detection (Benign, XSS, SQLi) using CNN
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random, os

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


class AttackDataset(Dataset):
    """Dataset chuyển text thành token ASCII để CNN xử lý"""
    def __init__(self, texts, labels, max_len=512, augment=False):
        self.texts, self.labels, self.max_len, self.augment = texts, labels, max_len, augment
    
    def __len__(self): return len(self.texts)
    
    def text_to_tokens(self, text):
        # Mỗi ký tự -> mã ASCII (0-255), padding = 0
        tokens = [ord(c) if ord(c) < 256 else 1 for c in str(text)[:self.max_len]]
        return tokens + [0] * (self.max_len - len(tokens))
    
    def augment_text(self, text):
        # Data augmentation: random swap case để tăng dữ liệu
        if random.random() < 0.3: return text
        return ''.join(c.swapcase() if random.random() < 0.1 and c.isalpha() else c for c in str(text))
    
    def __getitem__(self, idx):
        text = self.augment_text(self.texts[idx]) if self.augment else str(self.texts[idx])
        return torch.tensor(self.text_to_tokens(text), dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class ConvBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        # Conv1D với padding giữ nguyên kích thước
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.bn = nn.BatchNorm1d(out_ch)  # Batch Normalization giúp training ổn định
    
    def forward(self, x): return F.relu(self.bn(self.conv(x)))


class AttackNet(nn.Module):
    """
    CNN Multi-Scale cho Text Classification:
    - Embedding: chuyển token (0-255) thành vector 128D
    - Multi-scale Conv: dùng nhiều kernel size (2,3,4,5,7) để bắt n-gram features
    - 2 lớp Conv mỗi scale để học features phức tạp hơn
    - AdaptiveMaxPool: lấy feature quan trọng nhất từ mỗi scale
    - Classifier: 3 lớp FC với Dropout chống overfitting
    """
    def __init__(self, vocab_size=256, embed_dim=128, num_filters=128, num_classes=3, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        kernels = [2, 3, 4, 5, 7]  # Nhiều kernel size bắt các n-gram khác nhau
        self.convs1 = nn.ModuleList([ConvBlock(embed_dim, num_filters, k) for k in kernels])
        self.convs2 = nn.ModuleList([ConvBlock(num_filters, num_filters, k) for k in kernels])
        self.pool = nn.AdaptiveMaxPool1d(1)  # Lấy giá trị max từ toàn bộ sequence
        # Classifier với 3 FC layers
        total = num_filters * len(kernels)
        self.classifier = nn.Sequential(
            nn.Linear(total, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        # Kaiming init cho Conv, Xavier cho Linear - giúp training hội tụ nhanh
        for m in self.modules():
            if isinstance(m, nn.Conv1d): nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        x = self.embed(x).transpose(1, 2)  # [B, seq, emb] -> [B, emb, seq]
        # Multi-scale feature extraction
        outs = [self.pool(c2(c1(x))).squeeze(-1) for c1, c2 in zip(self.convs1, self.convs2)]
        return self.classifier(torch.cat(outs, dim=1))


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing: thay vì hard label [0,0,1], dùng soft label [0.05, 0.05, 0.9]
    Giúp model không quá tự tin, tăng generalization
    """
    def __init__(self, num_classes=3, smoothing=0.1):
        super().__init__()
        self.smoothing, self.num_classes = smoothing, num_classes
    
    def forward(self, pred, target):
        with torch.no_grad():
            dist = torch.zeros_like(pred).fill_(self.smoothing / (self.num_classes - 1))
            dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-dist * F.log_softmax(pred, dim=1), dim=1))


class AttackDetector:
    """Main class để train và predict"""
    LABELS = {0: "Benign", 1: "XSS", 2: "SQLi"}
    
    def __init__(self, max_len=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.model = AttackNet(num_classes=3).to(self.device)
        print(f"[AttackDetector] Device: {self.device} | Params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, df, test_size=0.15, val_size=0.15):
        # Chia train/val/test với stratified sampling giữ tỷ lệ các class
        train_val, test = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df['label'])
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=SEED, stratify=train_val['label'])
        self.train_ds = AttackDataset(train['text'].tolist(), train['label'].tolist(), self.max_len, augment=True)
        self.val_ds = AttackDataset(val['text'].tolist(), val['label'].tolist(), self.max_len)
        self.test_ds = AttackDataset(test['text'].tolist(), test['label'].tolist(), self.max_len)
        print(f"Data: Train={len(self.train_ds)} | Val={len(self.val_ds)} | Test={len(self.test_ds)}")
    
    def train(self, epochs=25, batch_size=64, lr=0.002, patience=6):
        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(self.val_ds, batch_size=batch_size*2, num_workers=0)
        
        # AdamW: Adam với weight decay (L2 regularization) chống overfitting
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # CosineAnnealing: giảm LR theo hình cosine, giúp thoát local minima
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
        criterion = LabelSmoothingLoss(num_classes=3, smoothing=0.1)
        
        best_acc, wait = 0, 0
        for ep in range(epochs):
            # Training
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for tokens, labels in tqdm(train_loader, desc=f"Ep {ep+1}/{epochs}", leave=False):
                tokens, labels = tokens.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                out = self.model(tokens)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping chống exploding
                optimizer.step()
                train_loss += loss.item()
                train_correct += (out.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
            
            # Validation
            self.model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for tokens, labels in val_loader:
                    tokens, labels = tokens.to(self.device), labels.to(self.device)
                    val_correct += (self.model(tokens).argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
            
            train_acc, val_acc = 100*train_correct/train_total, 100*val_correct/val_total
            print(f"Ep {ep+1:2d} | Train: {train_loss/len(train_loader):.4f} {train_acc:.2f}% | Val: {val_acc:.2f}%")
            scheduler.step()
            
            # Early stopping: dừng nếu không cải thiện sau `patience` epochs
            if val_acc > best_acc:
                best_acc, wait = val_acc, 0
                self.save('attack_model_best.pt')
            else:
                wait += 1
                if wait >= patience: print(f"Early stop at epoch {ep+1}"); break
        
        self.load('attack_model_best.pt')
        print(f"Training done! Best Val Acc: {best_acc:.2f}%")
    
    def evaluate(self):
        loader = DataLoader(self.test_ds, batch_size=128, num_workers=0)
        self.model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for tokens, labels in loader:
                preds.extend(self.model(tokens.to(self.device)).argmax(1).cpu().numpy())
                labels_all.extend(labels.numpy())
        print(f"\nTest Accuracy: {100*(np.array(preds)==np.array(labels_all)).mean():.2f}%")
        print(classification_report(labels_all, preds, target_names=list(self.LABELS.values()), digits=4))
    
    def predict(self, text):
        self.model.eval()
        tokens = [ord(c) if ord(c) < 256 else 1 for c in str(text)[:self.max_len]]
        tokens += [0] * (self.max_len - len(tokens))
        x = torch.tensor([tokens], dtype=torch.long).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            probs = F.softmax(out, dim=1)[0]
            pred = out.argmax(1).item()
        return {'label': self.LABELS[pred], 'label_id': pred, 'confidence': probs[pred].item(),
                'probabilities': {self.LABELS[i]: probs[i].item() for i in range(3)}}
    
    def save(self, path='attack_model.pt'):
        torch.save({'model': self.model.state_dict(), 'max_len': self.max_len}, path)
    
    def load(self, path='attack_model.pt'):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)


def main():
    # Load và chuẩn bị data
    print("Loading data...")
    xss_df = pd.read_csv('xss_data.csv')
    if 'Sentence' in xss_df.columns: xss_df = xss_df.rename(columns={'Sentence': 'text', 'Label': 'label'})
    xss_df = xss_df[['text', 'label']].dropna()
    xss_attack = xss_df[xss_df['label'] == 1].copy(); xss_attack['label'] = 1
    xss_benign = xss_df[xss_df['label'] == 0].copy(); xss_benign['label'] = 0
    
    sqli_df = pd.read_csv('sqli_data.csv')
    if 'Query' in sqli_df.columns: sqli_df = sqli_df.rename(columns={'Query': 'text', 'Label': 'label'})
    sqli_df = sqli_df[['text', 'label']].dropna()
    sqli_attack = sqli_df[sqli_df['label'] == 1].copy(); sqli_attack['label'] = 2  # SQLi = class 2
    sqli_benign = sqli_df[sqli_df['label'] == 0].copy(); sqli_benign['label'] = 0
    
    # Merge và shuffle
    df = pd.concat([xss_benign, sqli_benign, xss_attack, sqli_attack]).drop_duplicates(subset='text')
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Total: {len(df)} | Benign: {(df['label']==0).sum()} | XSS: {(df['label']==1).sum()} | SQLi: {(df['label']==2).sum()}")
    
    # Train
    detector = AttackDetector(max_len=512)
    detector.prepare_data(df)
    detector.train(epochs=25, batch_size=64, lr=0.002, patience=6)
    detector.evaluate()


if __name__ == "__main__":
    main()

