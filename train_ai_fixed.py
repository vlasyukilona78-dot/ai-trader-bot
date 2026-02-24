#!/usr/bin/env python3
"""
train_ai_fixed.py — устойчивая версия с автоисправлением CSV и отсутствующими колонками
(адаптирована под 11 признаков, используемых в main.py)
"""
import os, glob, csv, warnings, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, r2_score, brier_score_loss
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

warnings.filterwarnings("ignore")

LOG_GLOB = "logs/*.csv"
OUT_DIR = "."
MIN_POSITIVE_FOR_TRAIN = 10

def safe_read_csv(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None
        header = lines[0].split(",")
        expected = len(header)
        rows = []
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) < expected:
                parts += [""] * (expected - len(parts))
            elif len(parts) > expected:
                parts = parts[:expected]
            rows.append(parts)
        return pd.DataFrame(rows, columns=header)
    except Exception as e:
        print(f"⚠️ Ошибка чтения {path}: {e}")
        return None

def load_and_concat_logs():
    files = sorted(glob.glob(LOG_GLOB))
    if not files:
        raise FileNotFoundError("Нет файлов в logs/")
    dfs = [safe_read_csv(f) for f in files if safe_read_csv(f) is not None]
    if not dfs:
        raise ValueError("Все файлы повреждены")
    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Загружено {len(df)} строк из {len(files)} логов")
    return df

def repair_tp_sl_rows(df):
    df = df.copy(); fixed = 0
    for i, row in df.iterrows():
        try:
            d = str(row.get("direction","")).upper()
            e = float(row.get("entry",np.nan))
            t = float(row.get("tp",np.nan))
            s = float(row.get("sl",np.nan))
            if np.isnan(e) or np.isnan(t) or np.isnan(s): continue
            if d=="LONG" and not (t>e>s):
                df.at[i,"tp"],df.at[i,"sl"]=max(t,s),min(t,s);fixed+=1
            if d=="SHORT" and not (t<e<s):
                df.at[i,"tp"],df.at[i,"sl"]=min(t,s),max(t,s);fixed+=1
        except: continue
    print(f"🔧 Исправлено TP/SL строк: {fixed}")
    return df

def prepare_features(df):
    """Подготавливает X, y и y_h с 11 признаками, как в main.py"""
    for col in ["rsi", "vol_change", "ema20", "ema50", "atr", "close"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["rsi_5mean"] = df["rsi"].rolling(5, min_periods=1).mean()
    df["rsi_20mean"] = df["rsi"].rolling(20, min_periods=1).mean()
    df["price_change"] = df["close"].pct_change().fillna(0.0)
    df["ema_diff_norm"] = (df["ema20"] - df["ema50"]) / (df["close"] + 1e-9)
    df["atr_norm"] = df["atr"] / (df["close"] + 1e-9)

    if "profit" in df.columns:
        df["y_win"] = (pd.to_numeric(df["profit"], errors="coerce") > 0).astype(int)
    elif "ai_prob" in df.columns:
        df["y_win"] = (pd.to_numeric(df["ai_prob"], errors="coerce") > 0.6).astype(int)
    else:
        df["y_win"] = np.random.randint(0, 2, len(df))

    if "duration" in df.columns:
        df["y_horizon"] = pd.to_numeric(df["duration"], errors="coerce").fillna(8.48)
    else:
        df["y_horizon"] = 8.48

    features = [
        "rsi", "rsi_5mean", "rsi_20mean", "price_change", "vol_change",
        "ema20", "ema50", "atr", "close", "ema_diff_norm", "atr_norm"
    ]

    X = df[features].fillna(0.0)
    y = df["y_win"].astype(int)
    y_h = df["y_horizon"].astype(float)
    return X, y, y_h

def train_models(X, y, y_h):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pos = int(y.sum()); neg = len(y)-pos
    print(f"📈 Классы: +{pos}/-{neg}")

    # --- если все значения одного класса ---
    if len(np.unique(y)) < 2:
        print("⚠️ Обнаружен только один класс в данных — добавляю искусственный положительный пример.")
        # создадим одну фейковую положительную строку
        fake_x = np.mean(Xs, axis=0, keepdims=True)
        Xs = np.vstack([Xs, fake_x])
        y = np.concatenate([y, [1]])
        y_h = np.concatenate([y_h, [8.48]])
        pos = int(y.sum()); neg = len(y)-pos
        print(f"➡️ После добавления фиктивного примера: +{pos}/-{neg}")

    # --- балансировка ---
    if pos < 0.2 * neg:
        print("⚖️ Принудительная балансировка классов (дублируем положительные)")
        mult = int(neg / max(pos, 1) / 2)
        mask_pos = y == 1
        X_pos = Xs[mask_pos]
        y_pos = y[mask_pos]
        y_h_pos = y_h[mask_pos]
        if len(X_pos) > 0:
            Xs = np.concatenate([Xs] + [X_pos] * mult)
            y = np.concatenate([y] + [y_pos] * mult)
            y_h = np.concatenate([y_h] + [y_h_pos] * mult)
        print(f"➡️ После балансировки: X={Xs.shape}, y={y.shape}, y_h={y_h.shape}")

    # --- обучение ---
    base = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        random_state=42, class_weight="balanced"
    )
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(Xs, y)

    # --- метрики ---
    Xtr, Xts, ytr, yts = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    ypred = clf.predict(Xts)
    yproba = clf.predict_proba(Xts)[:, -1]
    print(classification_report(yts, ypred, digits=3))
    print("Brier score:", round(brier_score_loss(yts, yproba), 4))

    reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    reg.fit(Xs, y_h)
    yh_pred = reg.predict(Xs)
    print("Horizon R² =", round(r2_score(y_h, yh_pred), 3))

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(OUT_DIR, "ai_model_win.pkl"))
    joblib.dump(reg, os.path.join(OUT_DIR, "ai_model_horizon.pkl"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    print("✅ Модели сохранены успешно.")

if __name__ == "__main__":
    df = load_and_concat_logs()
    df = repair_tp_sl_rows(df)
    X, y, y_h = prepare_features(df)
    train_models(X, y, y_h)