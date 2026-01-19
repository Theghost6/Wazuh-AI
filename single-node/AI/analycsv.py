import pandas as pd
from sklearn.utils import resample

df = pd.read_csv('web_multi_labels.csv')

print(f"Original:  {len(df)}")

# Chỉ loại bỏ payload quá ngắn
df = df[df['payload'].str.len() >= 10]

print(f"After removing short:  {len(df)}")

# Cân bằng
benign = df[df['Label'] == 'BENIGN']
xss = df[df['Label'] == 'XSS']
sql = df[df['Label'] == 'SQL_INJECTION']

print(f"\nBefore balance:")
print(f"  BENIGN: {len(benign)}")
print(f"  XSS: {len(xss)}")
print(f"  SQL:  {len(sql)}")

target = 10000

benign_bal = resample(benign, n_samples=min(target, len(benign)), random_state=42, replace=True)
xss_bal = resample(xss, n_samples=min(target, len(xss)), random_state=42, replace=True)
sql_bal = resample(sql, n_samples=min(target, len(sql)), random_state=42, replace=True)

df_clean = pd.concat([benign_bal, xss_bal, sql_bal])
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAfter balance:")
print(f"  BENIGN: {len(benign_bal)}")
print(f"  XSS: {len(xss_bal)}")
print(f"  SQL:  {len(sql_bal)}")

df_clean.to_csv('web_multi_labels_balanced.csv', index=False)

print(f"\n✅ Total:  {len(df_clean)} samples saved")
