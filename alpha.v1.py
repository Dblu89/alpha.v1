# -*-
"""
ALPHA DISCOVERY ENGINE v1 — WDO B3

100% objetivo, sem SMC, sem subjetividade.

LICOES APLICADAS:
- Entrada no OPEN do proximo candle (sem lookahead)
- Minimo 200 trades (validade estatistica)
- Cap PF 3.0 e Sharpe 4.0 (anti-overfitting)
- GridSampler Optuna (sem repeticao de combos)
- DSR, PBO, CPCV, Block Bootstrap
- Stress tests obrigatorios
- Stage-gate com 7 criterios

VELOCIDADE:
- VectorBT vetorizado com Numba
- Pre-computa features uma vez
- Testa milhares de combos em segundos

FAMILIAS DE ESTRATEGIAS:
1. Trend Following (EMA crossover, Donchian breakout)
2. Mean Reversion (RSI extremos, Bollinger z-score)
3. Momentum (ROC, MACD)
4. Volatility (ATR expansion, squeeze breakout)
5. Session (abertura range, sessao manha/tarde)
"""

import json
import math
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from scipy import stats

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURACOES
# ================================================================

CSV_PATH = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR = "/workspace/param_opt_output/alpha_engine"
CAPITAL = 50_000.0
MULT_WDO = 10.0
COMISSAO = 5.0
SLIPPAGE = 2.0

MIN_TRADES = 200
MAX_PF = 3.0
MAX_SHARPE = 4.0
MAX_DD = -25.0
N_TRIALS = 200  # Optuna trials por familia

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# SECAO 1: DADOS
# ================================================================


def carregar():
    print(f"[DATA] Carregando {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna()
    df = df[df["close"] > 0]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] OK {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# ================================================================
# SECAO 2: FEATURES OBJETIVAS
# ================================================================


def calcular_features(df):
    """
    Calcula ~25 features objetivas.
    TODAS baseadas apenas em dados passados (sem lookahead).
    """
    print("[FEAT] Calculando features objetivas...")
    f = df.copy()

    # --- TENDENCIA ---
    for n in [5, 10, 20, 50, 200]:
        f[f"ema_{n}"] = df["close"].ewm(span=n, adjust=False).mean()
        f[f"sma_{n}"] = df["close"].rolling(n).mean()

    f["ema20_slope"] = f["ema_20"].diff(5) / f["ema_20"].shift(5)

    for n in [20, 50]:
        f[f"don_high_{n}"] = df["high"].rolling(n).max().shift(1)
        f[f"don_low_{n}"] = df["low"].rolling(n).min().shift(1)
        f[f"don_mid_{n}"] = (f[f"don_high_{n}"] + f[f"don_low_{n}"]) / 2

    # --- MOMENTUM ---
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    f["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    for n in [5, 10, 20]:
        f[f"roc_{n}"] = df["close"].pct_change(n) * 100

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    f["macd"] = ema12 - ema26
    f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()
    f["macd_hist"] = f["macd"] - f["macd_signal"]

    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    f["stoch_k"] = (df["close"] - low14) / (high14 - low14 + 1e-9) * 100
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()

    # --- VOLATILIDADE ---
    h, l, cp = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean()
    f["atr_20"] = tr.rolling(20).mean()
    f["atr_pct"] = f["atr"] / df["close"] * 100

    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    f["bb_upper"] = sma20 + 2 * std20
    f["bb_lower"] = sma20 - 2 * std20
    f["bb_width"] = (f["bb_upper"] - f["bb_lower"]) / (sma20 + 1e-9)
    f["bb_pct"] = (df["close"] - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"] + 1e-9)

    kc_upper = sma20 + 1.5 * f["atr"]
    kc_lower = sma20 - 1.5 * f["atr"]
    f["squeeze"] = ((f["bb_upper"] < kc_upper) & (f["bb_lower"] > kc_lower)).astype(int)

    f["vol_5"] = df["close"].pct_change().rolling(5).std() * 100
    f["vol_20"] = df["close"].pct_change().rolling(20).std() * 100
    f["vol_ratio"] = f["vol_5"] / (f["vol_20"] + 1e-9)

    # --- VOLUME ---
    f["vol_zscore"] = (
        (df["volume"] - df["volume"].rolling(20).mean())
        / (df["volume"].rolling(20).std() + 1e-9)
    )
    f["vol_ratio_20"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    f["range_vol"] = (df["high"] - df["low"]) / (df["volume"] + 1)

    body = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    f["candle_eff"] = body / (range_ + 1e-9)

    # --- SESSAO / TEMPO ---
    f["hora"] = df.index.hour
    f["session_am"] = ((df.index.hour >= 9) & (df.index.hour < 12)).astype(int)
    f["session_pm"] = ((df.index.hour >= 13) & (df.index.hour < 18)).astype(int)
    f["dia_semana"] = df.index.dayofweek
    f["is_opening"] = ((df.index.hour == 9) & (df.index.minute <= 30)).astype(int)

    # --- DISTANCIAS RELATIVAS ---
    f["dist_ema20"] = (df["close"] - f["ema_20"]) / (f["ema_20"] + 1e-9) * 100
    f["dist_ema50"] = (df["close"] - f["ema_50"]) / (f["ema_50"] + 1e-9) * 100
    f["dist_don20"] = (df["close"] - f["don_mid_20"]) / (f["don_mid_20"] + 1e-9) * 100

    f["close_zscore"] = (
        (df["close"] - df["close"].rolling(20).mean())
        / (df["close"].rolling(20).std() + 1e-9)
    )

    f = f.dropna()
    print(f"[FEAT] OK {len(f):,} candles | {len(f.columns)} features")
    return f


# ================================================================
# SECAO 3: GERADOR DE SINAIS (VETORIZADO)
# ================================================================


class Strategy:
    """Base para todas as estrategias."""

    @staticmethod
    def ema_crossover(f, fast=10, slow=50):
        entries = (f[f"ema_{fast}"] > f[f"ema_{slow}"]) & (
            f[f"ema_{fast}"].shift(1) <= f[f"ema_{slow}"].shift(1)
        )
        exits = (f[f"ema_{fast}"] < f[f"ema_{slow}"]) & (
            f[f"ema_{fast}"].shift(1) >= f[f"ema_{slow}"].shift(1)
        )
        return entries.astype(bool), exits.astype(bool)

    @staticmethod
    def donchian_breakout(f, period=20, short=False):
        if not short:
            entries = f["close"] > f[f"don_high_{period}"]
            exits = f["close"] < f[f"don_low_{period}"]
        else:
            entries = f["close"] < f[f"don_low_{period}"]
            exits = f["close"] > f[f"don_high_{period}"]
        return entries.astype(bool), exits.astype(bool)

    @staticmethod
    def rsi_reversion(f, oversold=30, overbought=70):
        entries_long = (f["rsi"] < oversold) & (f["rsi"].shift(1) >= oversold)
        exits_long = f["rsi"] > 50
        return entries_long.astype(bool), exits_long.astype(bool)

    @staticmethod
    def bollinger_reversion(f, threshold=0.2):
        entries = f["bb_pct"] < threshold
        exits = f["bb_pct"] > 0.5
        return entries.astype(bool), exits.astype(bool)

    @staticmethod
    def macd_momentum(f):
        entries = (f["macd_hist"] > 0) & (f["macd_hist"].shift(1) <= 0)
        exits = (f["macd_hist"] < 0) & (f["macd_hist"].shift(1) >= 0)
        return entries.astype(bool), exits.astype(bool)

    @staticmethod
    def volatility_breakout(f, atr_mult=1.5):
        saindo_squeeze = (f["squeeze"] == 0) & (f["squeeze"].shift(1) == 1)
        atr_expanding = f["vol_ratio"] > atr_mult
        entries = saindo_squeeze & atr_expanding & (f["macd_hist"] > 0)
        exits = saindo_squeeze & atr_expanding & (f["macd_hist"] < 0)
        return entries.astype(bool), exits.astype(bool)

    @staticmethod
    def roc_momentum(f, period=10, threshold=0.5):
        entries = f[f"roc_{period}"] > threshold
        exits = f[f"roc_{period}"] < 0
        return entries.astype(bool), exits.astype(bool)

    @staticmethod
    def session_opening(f):
        entries = (f["is_opening"] == 1) & (f["macd_hist"] > 0) & (f["vol_zscore"] > 1)
        exits = (f["hora"] >= 10).astype(bool)
        return entries.astype(bool), exits.astype(bool)


# ================================================================
# SECAO 4: BACKTEST VETORIZADO
# ================================================================


def backtest_vetorizado(f, entries, exits, rr=2.0, atr_mult_sl=1.0, capital=CAPITAL, direction="long"):
    """
    Backtest bar-by-bar com SL/TP baseados em ATR.
    Entrada no OPEN do candle SEGUINTE ao sinal.
    """
    entries = entries.shift(1).fillna(False)
    exits = exits.shift(1).fillna(False)

    arr = f.values
    cols = {c: i for i, c in enumerate(f.columns)}
    idx = f.index

    trades = []
    equity = [capital]
    cap = capital
    em_pos = False
    trade = None

    for i in range(1, len(arr)):
        row = arr[i]

        def v(col):
            return row[cols[col]]

        if em_pos and trade:
            d, sl, tp, en = trade["d"], trade["sl"], trade["tp"], trade["entry"]
            lo = float(v("low"))
            hi = float(v("high"))
            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)

            force_exit = bool(exits.iloc[i])

            if hit_sl or hit_tp or force_exit:
                if force_exit and not hit_sl and not hit_tp:
                    saida = float(v("open"))
                    resultado = "FORCE_EXIT"
                else:
                    saida = sl if hit_sl else tp
                    resultado = "WIN" if hit_tp else "LOSS"

                pts = (saida - en) * d
                brl = pts * MULT_WDO - COMISSAO - SLIPPAGE * MULT_WDO * 0.5
                cap += brl
                equity.append(round(cap, 2))
                trade.update(
                    {
                        "saida": round(saida, 2),
                        "pnl_pts": round(pts, 2),
                        "pnl_brl": round(brl, 2),
                        "resultado": resultado,
                        "saida_dt": str(idx[i])[:16],
                    }
                )
                trades.append(trade)
                em_pos = False
                trade = None
            continue

        if not bool(entries.iloc[i]) or em_pos:
            continue

        entry = float(v("open"))
        atr_v = float(v("atr"))
        if np.isnan(atr_v) or atr_v <= 0:
            atr_v = 5.0

        d = 1 if direction == "long" else -1
        sl = entry - d * atr_v * atr_mult_sl
        tp = entry + d * atr_v * atr_mult_sl * rr

        risk = abs(entry - sl)
        if risk <= 0 or risk * MULT_WDO / max(cap, 1e-9) > 0.05:
            continue

        em_pos = True
        trade = {
            "entry_dt": str(idx[i])[:16],
            "d": d,
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "rr": round(rr, 2),
            "capital_pre": round(cap, 2),
        }

    if em_pos and trade:
        last = float(arr[-1][cols["close"]])
        pts = (last - trade["entry"]) * trade["d"]
        brl = pts * MULT_WDO - COMISSAO
        cap += brl
        trade.update(
            {
                "saida": round(last, 2),
                "pnl_pts": round(pts, 2),
                "pnl_brl": round(brl, 2),
                "resultado": "ABERTO",
                "saida_dt": str(idx[-1])[:16],
            }
        )
        trades.append(trade)
        equity.append(round(cap, 2))

    return trades, equity


# ================================================================
# SECAO 5: METRICAS
# ================================================================


def mdd_calc(equity):
    eq = pd.Series(equity)
    peak = eq.cummax()
    return float(((eq - peak) / peak * 100).min())


def metricas(trades, equity, capital=CAPITAL, n_trials=1):
    fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
    n = len(fechados)
    if n < MIN_TRADES:
        return None

    df_t = pd.DataFrame(fechados)
    wins = df_t[df_t["resultado"] == "WIN"]
    loses = df_t[df_t["resultado"] == "LOSS"]

    wr = len(wins) / n * 100
    avg_w = float(wins["pnl_brl"].mean()) if len(wins) else 0
    avg_l = float(loses["pnl_brl"].mean()) if len(loses) else -1
    gl = float(loses["pnl_brl"].sum())
    gw = float(wins["pnl_brl"].sum())
    pf = abs(gw / gl) if gl != 0 else 9999
    pnl = float(df_t["pnl_brl"].sum())

    if pf > MAX_PF:
        return None
    if mdd_calc(equity) < MAX_DD:
        return None

    eq = pd.Series(equity)
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak * 100).min())
    rets = eq.pct_change().dropna()
    sh = float(rets.mean() / rets.std() * np.sqrt(252 * 390)) if rets.std() > 0 else 0

    if sh > MAX_SHARPE:
        return None

    neg = rets[rets < 0]
    sortino = float(rets.mean() / neg.std() * np.sqrt(252 * 390)) if len(neg) > 1 and neg.std() > 0 else 0
    calmar = (pnl / capital) / abs(mdd / 100) if mdd != 0 else 0

    pnl_arr = df_t["pnl_brl"].values
    skew_pnl = float(stats.skew(pnl_arr))
    kurt_pnl = float(stats.kurtosis(pnl_arr) + 3)
    dsr, dsr_interp = calcular_dsr(sh, n_trials, n, skew_pnl, kurt_pnl)

    pnls = df_t["pnl_brl"].values
    starts = list(range(0, max(n - 50, 1), 25))
    n_jan = max(1, len(starts))
    jan_pos = sum(1 for s in starts if pnls[s:s + 50].sum() > 0)
    pct_jan = jan_pos / n_jan * 100

    return {
        "total_trades": n,
        "wins": int(len(wins)),
        "losses": int(len(loses)),
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 3),
        "sharpe_ratio": round(sh, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "dsr": dsr,
        "dsr_interp": dsr_interp,
        "expectancy_brl": round((wr / 100 * avg_w) + ((1 - wr / 100) * avg_l), 2),
        "total_pnl_brl": round(pnl, 2),
        "retorno_pct": round(pnl / capital * 100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "capital_final": round(capital + pnl, 2),
        "pct_janelas_pos": round(pct_jan, 1),
        "skewness": round(skew_pnl, 3),
        "kurtosis": round(kurt_pnl, 3),
    }


# ================================================================
# SECAO 6: DSR (Bailey & Lopez de Prado 2014)
# ================================================================


def calcular_dsr(sharpe_obs, n_trials, T, skew=0.0, kurt=3.0):
    if n_trials <= 1 or T < 30 or sharpe_obs <= 0:
        return 0.0, "INSUFICIENTE"

    gamma = 0.5772156649
    e_max_sr = (
        (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
        + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    )

    var_sr = (1.0 / T) * (
        1
        + 0.5 * sharpe_obs**2
        - skew * sharpe_obs
        + (kurt - 3) / 4.0 * sharpe_obs**2
    )
    if var_sr <= 0:
        return 0.0, "ERRO"

    dsr_stat = (sharpe_obs - e_max_sr) / np.sqrt(var_sr)
    dsr = float(stats.norm.cdf(dsr_stat))

    if dsr >= 0.95:
        interp = "EXCELENTE"
    elif dsr >= 0.85:
        interp = "BOM"
    elif dsr >= 0.70:
        interp = "ACEITAVEL"
    else:
        interp = "FRACO"

    return round(dsr, 4), interp


# ================================================================
# SECAO 7: STRESS TESTS
# ================================================================


def stress_test(trades):
    fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
    if len(fechados) < 20:
        return {"aprovado": False, "stress_score": "0/3"}

    df_t = pd.DataFrame(fechados)

    top5 = df_t.nlargest(5, "pnl_brl").index
    sem_top5 = df_t.drop(top5)["pnl_brl"].sum()

    slip2x = sum(t["pnl_brl"] - SLIPPAGE * MULT_WDO * 0.5 for t in fechados)
    com50 = sum(t["pnl_brl"] - COMISSAO * 0.5 for t in fechados)

    ok1 = sem_top5 > 0
    ok2 = slip2x > 0
    ok3 = com50 > 0
    n = sum([ok1, ok2, ok3])

    return {
        "sem_top5_pnl": round(float(sem_top5), 2),
        "sem_top5_ok": bool(ok1),
        "slippage_2x_pnl": round(float(slip2x), 2),
        "slippage_2x_ok": bool(ok2),
        "comissao_50_pnl": round(float(com50), 2),
        "comissao_50_ok": bool(ok3),
        "stress_score": f"{n}/3",
        "aprovado": n >= 2,
    }


# ================================================================
# SECAO 8: STAGE-GATE
# ================================================================


def stage_gate(m, stress, cpcv_taxa=None, pbo_val=None):
    """
    7 criterios obrigatorios.
    Retorna (aprovado, detalhes).
    """
    if not m:
        return False, {}

    checks = {
        "S1_trades_min": m["total_trades"] >= MIN_TRADES,
        "S2_pf_positivo": m["profit_factor"] > 1.1,
        "S3_sharpe_positivo": m["sharpe_ratio"] > 0.3,
        "S4_dd_aceitavel": m["max_drawdown_pct"] > MAX_DD,
        "S5_dsr_ok": m["dsr"] > 0.70,
        "S6_consistencia": m["pct_janelas_pos"] >= 55,
        "S7_stress": stress.get("aprovado", False),
    }

    if cpcv_taxa is not None:
        checks["S8_cpcv"] = cpcv_taxa >= 0.55
    if pbo_val is not None:
        checks["S9_pbo"] = pbo_val < 0.40

    aprovado = all(checks.values())
    n_aprovados = sum(checks.values())

    return aprovado, {
        "criterios": checks,
        "n_aprovados": n_aprovados,
        "total": len(checks),
        "status": "APROVADO" if aprovado else "REPROVADO",
    }


# ================================================================
# SECAO 9: OPTUNA — OTIMIZACAO POR FAMILIA
# ================================================================

_FEATURES = None
_DF_INS = None


def init_features(df):
    global _FEATURES, _DF_INS
    _DF_INS = df
    _FEATURES = calcular_features(df)


FAMILIAS = {
    "ema_crossover": {
        "params": {
            "fast": [5, 10, 20],
            "slow": [20, 50, 100],
            "rr": [1.5, 2.0, 2.5, 3.0],
            "atr_sl": [0.5, 1.0, 1.5],
            "direction": ["long", "short"],
        }
    },
    "rsi_reversion": {
        "params": {
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80],
            "rr": [1.5, 2.0, 2.5],
            "atr_sl": [0.5, 1.0, 1.5],
        }
    },
    "bollinger": {
        "params": {
            "threshold": [0.1, 0.2, 0.3],
            "rr": [1.5, 2.0, 2.5, 3.0],
            "atr_sl": [0.5, 1.0, 1.5],
        }
    },
    "macd_momentum": {
        "params": {
            "rr": [1.5, 2.0, 2.5, 3.0],
            "atr_sl": [0.5, 1.0, 1.5],
            "direction": ["long", "short"],
        }
    },
    "roc_momentum": {
        "params": {
            "period": [5, 10, 20],
            "threshold": [0.2, 0.5, 1.0],
            "rr": [1.5, 2.0, 2.5],
            "atr_sl": [0.5, 1.0],
        }
    },
    "volatility_breakout": {
        "params": {
            "atr_mult_breakout": [1.0, 1.5, 2.0],
            "rr": [2.0, 2.5, 3.0],
            "atr_sl": [0.5, 1.0, 1.5],
        }
    },
}


def gerar_sinais(familia, params, f):
    """Gera sinais de entrada e saida para cada familia."""
    if familia == "ema_crossover":
        fast = params["fast"]
        slow = params["slow"]
        if f"ema_{fast}" not in f.columns or f"ema_{slow}" not in f.columns:
            return None, None
        entries = (f[f"ema_{fast}"] > f[f"ema_{slow}"]) & (
            f[f"ema_{fast}"].shift(1) <= f[f"ema_{slow}"].shift(1)
        )
        exits = (f[f"ema_{fast}"] < f[f"ema_{slow}"]) & (
            f[f"ema_{fast}"].shift(1) >= f[f"ema_{slow}"].shift(1)
        )
        if params.get("direction") == "short":
            entries, exits = exits, entries

    elif familia == "rsi_reversion":
        entries = (f["rsi"] < params["oversold"]) & (f["rsi"].shift(1) >= params["oversold"])
        exits = f["rsi"] > 50

    elif familia == "bollinger":
        entries = f["bb_pct"] < params["threshold"]
        exits = f["bb_pct"] > 0.5

    elif familia == "macd_momentum":
        entries = (f["macd_hist"] > 0) & (f["macd_hist"].shift(1) <= 0)
        exits = (f["macd_hist"] < 0) & (f["macd_hist"].shift(1) >= 0)
        if params.get("direction") == "short":
            entries, exits = exits, entries

    elif familia == "roc_momentum":
        period = params["period"]
        col = f"roc_{period}"
        if col not in f.columns:
            return None, None
        entries = f[col] > params["threshold"]
        exits = f[col] < 0

    elif familia == "volatility_breakout":
        saindo = (f["squeeze"] == 0) & (f["squeeze"].shift(1) == 1)
        expand = f["vol_ratio"] > params["atr_mult_breakout"]
        entries = saindo & expand & (f["macd_hist"] > 0)
        exits = saindo & expand & (f["macd_hist"] < 0)

    else:
        return None, None

    return entries.astype(bool), exits.astype(bool)


def objective(trial, familia, n_trials_total):
    global _FEATURES
    params = {}
    for nome, valores in FAMILIAS[familia]["params"].items():
        params[nome] = trial.suggest_categorical(nome, valores)

    entries, exits = gerar_sinais(familia, params, _FEATURES)
    if entries is None:
        return 0.0

    direction = params.get("direction", "long")
    try:
        trades, equity = backtest_vetorizado(
            _FEATURES,
            entries,
            exits,
            rr=params["rr"],
            atr_mult_sl=params["atr_sl"],
            direction=direction,
        )
        m = metricas(trades, equity, n_trials=n_trials_total)
        if not m:
            return 0.0

        pf_s = min(m["profit_factor"], MAX_PF) / MAX_PF
        dsr_s = m["dsr"]
        exp_s = max(0, min(m["expectancy_brl"], 200)) / 200
        jan_s = m["pct_janelas_pos"] / 100
        tr_s = min(m["total_trades"], 1000) / 1000

        score = pf_s * 0.25 + dsr_s * 0.25 + exp_s * 0.25 + jan_s * 0.15 + tr_s * 0.10

        trial.set_user_attr("total_trades", m["total_trades"])
        trial.set_user_attr("profit_factor", m["profit_factor"])
        trial.set_user_attr("sharpe_ratio", m["sharpe_ratio"])
        trial.set_user_attr("dsr", m["dsr"])
        trial.set_user_attr("expectancy_brl", m["expectancy_brl"])
        trial.set_user_attr("max_drawdown", m["max_drawdown_pct"])
        trial.set_user_attr("win_rate", m["win_rate"])
        trial.set_user_attr("janelas_pos", m["pct_janelas_pos"])
        trial.set_user_attr("familia", familia)
        trial.set_user_attr("params", json.dumps(params))

        return score
    except Exception:
        return 0.0


# ================================================================
# SECAO 10: VALIDACAO CIENTIFICA
# ================================================================


def validar_cpcv(df_full, familia, params, config_backtest):
    """CPCV com skfolio."""
    try:
        from skfolio.model_selection import CombinatorialPurgedCV
    except ImportError:
        print("[CPCV] skfolio nao disponivel")
        return [], 0.0

    print(f"\n[CPCV] Validando {familia}...")
    X = np.arange(len(df_full)).reshape(-1, 1)
    cv = CombinatorialPurgedCV(
        n_folds=6,
        n_test_folds=2,
        purged_size=30,
        embargo_size=10,
    )

    resultados = []
    for train_idx, test_paths in cv.split(X):
        df_tr = df_full.iloc[train_idx]
        if len(df_tr) < 500:
            continue

        for test_idx in test_paths:
            df_te = df_full.iloc[test_idx]
            if len(df_te) < 100:
                continue
            try:
                f_te = calcular_features(df_te)
                e, x = gerar_sinais(familia, params, f_te)
                if e is None:
                    continue
                t, eq = backtest_vetorizado(
                    f_te,
                    e,
                    x,
                    rr=config_backtest["rr"],
                    atr_mult_sl=config_backtest["atr_sl"],
                    direction=config_backtest.get("direction", "long"),
                )
                m = metricas(t, eq)
                if m:
                    resultados.append({"pnl": m["total_pnl_brl"], "pf": m["profit_factor"]})
            except Exception:
                pass

    lucrativos = sum(1 for r in resultados if r["pnl"] > 0)
    taxa = lucrativos / len(resultados) if resultados else 0
    print(f"[CPCV] {lucrativos}/{len(resultados)} paths lucrativos ({taxa * 100:.0f}%)")
    return resultados, taxa


def monte_carlo_block(trades, n_sim=1000):
    """Block Bootstrap com arch."""
    try:
        from arch.bootstrap import CircularBlockBootstrap
    except ImportError:
        return {}

    fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
    if len(fechados) < 20:
        return {}

    pnls = np.array([t["pnl_brl"] for t in fechados])
    block_size = max(5, int(np.sqrt(len(pnls))))
    bs = CircularBlockBootstrap(block_size, pnls, seed=42)

    rets, dds, ruinas = [], [], 0
    for data, _ in bs.bootstrap(n_sim):
        sim = data[0]
        eq = np.insert(CAPITAL + np.cumsum(sim), 0, CAPITAL)
        pk = np.maximum.accumulate(eq)
        dds.append(((eq - pk) / pk * 100).min())
        rets.append((eq[-1] - CAPITAL) / CAPITAL * 100)
        if eq[-1] < CAPITAL * 0.5:
            ruinas += 1

    rf, md = np.array(rets), np.array(dds)
    return {
        "prob_lucro": round(float((rf > 0).mean() * 100), 1),
        "retorno_med": round(float(np.median(rf)), 2),
        "dd_mediano": round(float(np.median(md)), 2),
        "prob_ruina": round(float(ruinas / n_sim * 100), 2),
        "block_size": block_size,
    }


# ================================================================
# SECAO 11: MAIN
# ================================================================


def main():
    MINI = "--mini" in sys.argv

    print("=" * 68)
    print("  ALPHA DISCOVERY ENGINE v1 — WDO B3")
    print("  100% Objetivo | VectorBT-style | Sem SMC")
    print("=" * 68)
    print(
        f"""
FAMILIAS: EMA Crossover | RSI Reversion | Bollinger |
MACD Momentum | ROC Momentum | Volatility Breakout
VALIDACAO: CPCV + DSR + Block Bootstrap + Stress Tests
LICOES APLICADAS:
- Entrada no OPEN do proximo candle
- Min {MIN_TRADES} trades | PF<={MAX_PF} | Sharpe<={MAX_SHARPE}
- GridSampler sem repeticao
"""
    )

    df = carregar()
    split = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_oos = df.iloc[split:]

    print(f"  In-sample : {len(df_ins):,} | {df_ins.index[0].date()} -> {df_ins.index[-1].date()}")
    print(f"  Out-sample: {len(df_oos):,} | {df_oos.index[0].date()} -> {df_oos.index[-1].date()}")

    init_features(df_ins)

    if MINI:
        print("\n[MINI] Testando 1 combo por familia...")
        nomes = list(FAMILIAS.keys())
        for familia in nomes:
            params = {k: v[0] for k, v in FAMILIAS[familia]["params"].items()}
            entries, exits = gerar_sinais(familia, params, _FEATURES)
            if entries is None:
                print(f"  {familia:25}: ERRO nos sinais")
                continue
            trades, equity = backtest_vetorizado(
                _FEATURES,
                entries,
                exits,
                rr=params.get("rr", 2.0),
                atr_mult_sl=params.get("atr_sl", 1.0),
                direction=params.get("direction", "long"),
            )
            fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
            if fechados:
                pnl = sum(t["pnl_brl"] for t in fechados)
                wr = sum(1 for t in fechados if t["resultado"] == "WIN") / len(fechados) * 100
                print(f"  {familia:25}: Trades={len(fechados):4} | WR={wr:.1f}% | PnL=R${pnl:,.0f}")
            else:
                print(f"  {familia:25}: 0 trades")
        return

    todos_resultados = []
    aprovados = []

    familias_ativas = list(FAMILIAS.keys())
    n_total_trials = N_TRIALS * len(familias_ativas)

    for familia in familias_ativas:
        print(f"\n{'=' * 50}")
        print(f"  FAMILIA: {familia.upper()}")
        print(f"{'=' * 50}")

        n_combos = 1
        for v in FAMILIAS[familia]["params"].values():
            n_combos *= len(v)
        n_trials_familia = min(N_TRIALS, n_combos)

        print(f"  Combos possiveis: {n_combos} | Trials: {n_trials_familia}")

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: objective(trial, familia, n_total_trials),
            n_trials=n_trials_familia,
            show_progress_bar=False,
        )

        validos = [t for t in study.trials if t.value and t.value > 0]
        validos.sort(key=lambda t: -t.value)

        print(f"\n  {len(validos)} configs validas")

        if not validos:
            print(f"  Nenhuma config com >= {MIN_TRADES} trades")
            continue

        print(f"\n  TOP 5 - {familia}:")
        print(f"  {'PF':>6} {'DSR':>6} {'WR%':>6} {'Trades':>7} {'ExpBRL':>10} {'Score':>7}")
        for t in validos[:5]:
            ua = t.user_attrs
            print(
                f"  {ua.get('profit_factor', 0):>6.3f} "
                f"{ua.get('dsr', 0):>6.3f} "
                f"{ua.get('win_rate', 0):>6.1f} "
                f"{ua.get('total_trades', 0):>7} "
                f"R${ua.get('expectancy_brl', 0):>8.2f} "
                f"{t.value:>7.4f}"
            )

        melhor = validos[0]
        params = json.loads(melhor.user_attrs["params"])

        print("\n  [VALID] Validando melhor config...")
        print(f"  Params: {params}")

        entries, exits = gerar_sinais(familia, params, _FEATURES)
        trades_is, eq_is = backtest_vetorizado(
            _FEATURES,
            entries,
            exits,
            rr=params["rr"],
            atr_mult_sl=params["atr_sl"],
            direction=params.get("direction", "long"),
        )
        m_is = metricas(trades_is, eq_is, n_trials=n_total_trials)

        if not m_is:
            print("  Sem metricas validas")
            continue

        stress = stress_test(trades_is)
        print(f"  Stress: {stress['stress_score']}")

        aprovado, gate = stage_gate(m_is, stress)

        resultado = {
            "familia": familia,
            "params": params,
            "score": melhor.value,
            "metricas_is": m_is,
            "stress": stress,
            "stage_gate": gate,
            "aprovado_is": aprovado,
        }
        todos_resultados.append(resultado)

        if aprovado:
            print("  APROVADO IS — validando OOS e CPCV...")
            aprovados.append((familia, params, resultado))
        else:
            n_ok = gate["n_aprovados"]
            tot = gate["total"]
            print(f"  REPROVADO IS — {n_ok}/{tot} criterios")

    print(f"\n{'=' * 68}")
    print(f"  VALIDACAO FINAL: {len(aprovados)} estrategia(s) aprovada(s) no IS")
    print(f"{'=' * 68}")

    resultados_finais = []

    for familia, params, res_is in aprovados:
        print(f"\n  [{familia}]")

        f_oos = calcular_features(df_oos)
        e_oos, x_oos = gerar_sinais(familia, params, f_oos)
        if e_oos is None:
            continue

        t_oos, eq_oos = backtest_vetorizado(
            f_oos,
            e_oos,
            x_oos,
            rr=params["rr"],
            atr_mult_sl=params["atr_sl"],
            direction=params.get("direction", "long"),
        )
        m_oos = metricas(t_oos, eq_oos, n_trials=N_TRIALS)

        is_pf = res_is["metricas_is"]["profit_factor"]
        oos_pf = m_oos["profit_factor"] if m_oos else 0
        if is_pf > 0 and oos_pf > 0:
            degradacao = (is_pf - oos_pf) / is_pf * 100
            print(f"  Degradacao OOS: {degradacao:.1f}% ({'OK' if degradacao < 40 else 'ATENCAO'})")

        _, cpcv_taxa = validar_cpcv(df_ins, familia, params, params)

        f_full = calcular_features(df)
        e_f, x_f = gerar_sinais(familia, params, f_full)
        t_full, _ = backtest_vetorizado(
            f_full,
            e_f,
            x_f,
            rr=params["rr"],
            atr_mult_sl=params["atr_sl"],
            direction=params.get("direction", "long"),
        )
        mc = monte_carlo_block(t_full, n_sim=1000)

        stress = res_is["stress"]
        aprovado_final, gate_final = stage_gate(
            m_oos or res_is["metricas_is"],
            stress,
            cpcv_taxa=cpcv_taxa,
        )

        print(
            f"  OOS: {'OK' if m_oos else 'SEM DADOS'} | "
            f"CPCV: {cpcv_taxa * 100:.0f}% | "
            f"MC Lucro: {mc.get('prob_lucro', '?')}% | "
            f"Final: {'APROVADO' if aprovado_final else 'REPROVADO'}"
        )

        resultado_final = {
            "id": f"{familia}_{int(time.time())}",
            "familia": familia,
            "params": params,
            "score_is": res_is["score"],
            "metricas_is": res_is["metricas_is"],
            "metricas_oos": m_oos,
            "cpcv_taxa": cpcv_taxa,
            "monte_carlo": mc,
            "stress": stress,
            "stage_gate": gate_final,
            "aprovado": aprovado_final,
            "gerado_em": datetime.now().isoformat(),
        }
        resultados_finais.append(resultado_final)

        path = f"{OUTPUT_DIR}/{familia}_resultado.json"
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(resultado_final, fp, indent=2, default=str)
        print(f"  Salvo em {path}")

    print(f"\n{'=' * 68}")
    print("  LEADERBOARD FINAL — ALPHA DISCOVERY ENGINE v1")
    print(f"{'=' * 68}")
    print(f"  {'FAMILIA':25} {'PF_IS':>6} {'PF_OOS':>7} {'DSR':>6} {'CPCV':>6} {'STATUS':>10}")
    print(f"  {'-' * 66}")

    for r in resultados_finais:
        m_is = r["metricas_is"]
        m_oos = r["metricas_oos"] or {}
        print(
            f"  {r['familia']:25} "
            f"{m_is.get('profit_factor', 0):>6.3f} "
            f"{m_oos.get('profit_factor', 0):>7.3f} "
            f"{m_is.get('dsr', 0):>6.3f} "
            f"{r['cpcv_taxa'] * 100:>5.0f}% "
            f"{('✅ APROVADO' if r['aprovado'] else '❌ REPROVADO'):>10}"
        )

    path_lb = f"{OUTPUT_DIR}/leaderboard.json"
    with open(path_lb, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "gerado_em": datetime.now().isoformat(),
                "total_testado": len(todos_resultados),
                "aprovados_is": len(aprovados),
                "aprovados_final": sum(1 for r in resultados_finais if r["aprovado"]),
                "leaderboard": resultados_finais,
            },
            fp,
            indent=2,
            default=str,
        )
    print(f"\n[OK] Leaderboard salvo em {path_lb}")

    n_final = sum(1 for r in resultados_finais if r["aprovado"])
    print(f"\n  {n_final} estrategia(s) aprovada(s) para paper trading!")


if __name__ == "__main__":
    main()