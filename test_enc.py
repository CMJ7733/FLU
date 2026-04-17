import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

path = r"D:\edison\EDISON\大三\数学建模\流感\data\raw\VIW_FNT.csv"

# 1. BOM 检测
with open(path, 'rb') as f:
    bom = f.read(4)
print("BOM bytes:", bom.hex())
if bom[:3] == b'\xef\xbb\xbf':
    print("=> UTF-8 with BOM")
elif bom[:2] == b'\xff\xfe':
    print("=> UTF-16 LE")
elif bom[:2] == b'\xfe\xff':
    print("=> UTF-16 BE")
else:
    print("=> No BOM")

# 2. pandas 编码尝试
import pandas as pd
chosen_enc = None
for enc in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
    try:
        df = pd.read_csv(path, encoding=enc, nrows=5, low_memory=False)
        print(f"\n[OK] encoding={enc}, cols={len(df.columns)}")
        print("cols[:8]:", list(df.columns[:8]))
        chosen_enc = enc
        break
    except Exception as e:
        print(f"[FAIL] {enc}: {type(e).__name__}: {str(e)[:100]}")

# 3. 全量 CHN 行数
if chosen_enc:
    print(f"\n统计 CHN 行数 (encoding={chosen_enc})...")
    n = 0
    for chunk in pd.read_csv(path, encoding=chosen_enc, chunksize=100000, low_memory=False):
        n += int((chunk['COUNTRY_CODE'] == 'CHN').sum())
    print(f"CHN 总行数: {n}")
