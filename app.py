import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Gold VWAP Reversion Sniper (GLD)",
    page_icon="🟡",
    layout="wide"
)

# ==========================================
# BARRA LATERAL: PARÁMETROS CUANTITATIVOS
# ==========================================
st.sidebar.title("🟡 Control Cuantitativo Oro")
st.sidebar.markdown("---")

ticker = "GLD"
st.sidebar.info(f"**Activo Institucional:** `{ticker}` (SPDR Gold Trust)")

days_back = st.sidebar.slider(
    "Días de Histórico (Velas 5m):",
    min_value=5,
    max_value=59,
    value=30,
    step=1
)

st.sidebar.markdown("### 🎯 Parámetros de Reversión")
z_threshold = st.sidebar.number_input("Umbral Z-Score (Entrada):", value=-1.00, step=0.05)
vol_multiplier = st.sidebar.number_input("Volumen Mínimo (vs Media 20):", value=1.10, step=0.05)

st.sidebar.markdown("### 🛡️ Reglas de Riesgo")
sl_pct = st.sidebar.number_input("Stop Loss Fijo (%):", value=0.50, step=0.05) / 100
tp_min_pct = st.sidebar.number_input("Take Profit Mínimo (%):", value=0.80, step=0.05) / 100

st.sidebar.button("🔄 Actualizar Datos de Mercado")

# ==========================================
# CARGA ROBUSTA DE DATOS (CON FALLBACK)
# ==========================================
@st.cache_data(ttl=120)
def load_gold_data(sym, days):
    period_str = f"{min(days, 59)}d"
    
    # Intento 1: yf.Ticker().history() (Más robusto en Streamlit Cloud)
    try:
        t = yf.Ticker(sym)
        raw = t.history(period=period_str, interval="5m", auto_adjust=False)
    except Exception:
        raw = pd.DataFrame()
        
    # Intento 2: yf.download() como fallback
    if raw.empty:
        try:
            raw = yf.download(sym, period=period_str, interval="5m", progress=False, auto_adjust=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
        except Exception:
            raw = pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame(index=raw.index)
    df['open'] = raw['Open']
    df['high'] = raw['High']
    df['low'] = raw['Low']
    df['close'] = raw['Close']
    df['volume'] = raw['Volume']
    df = df.dropna()

    if df.empty:
        return pd.DataFrame()

    # Normalización horaria a Nueva York (EST)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')

    df['date'] = df.index.date
    df['time_min'] = df.index.hour * 60 + df.index.minute

    # Ventana de Alta Liquidez: 08:00 a 13:30 EST (480 a 810 min)
    df['in_rth'] = (df['time_min'] >= 480) & (df['time_min'] <= 810)

    # 1. VWAP Acumulado de Sesión
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['tp'] * df['volume']
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / (df['cum_vol'] + 1e-6)

    # 2. Desviación Estándar Acumulada y Z-Score
    df['dev_sq'] = (df['close'] - df['vwap'])**2
    df['cum_dev_sq'] = df.groupby('date')['dev_sq'].cumsum()
    df['count_bars'] = df.groupby('date').cumcount() + 1
    df['vwap_std'] = np.sqrt(df['cum_dev_sq'] / df['count_bars'])
    df['vwap_lower_10'] = df['vwap'] - (abs(z_threshold) * df['vwap_std'])
    df['vwap_zscore'] = (df['close'] - df['vwap']) / (df['vwap_std'] + 1e-6)

    # 3. Absorción de Volumen
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_surge'] = df['volume'] > (df['vol_ma20'] * vol_multiplier)
    df['is_green'] = df['close'] > df['open']

    # Condición de Entrada
    cond_entry = (
        (df['vwap_zscore'] < z_threshold) &
        (df['is_green']) &
        (df['vol_surge']) &
        (df['in_rth'])
    )
    df['signal'] = np.where(cond_entry, 1, 0)
    return df

df = load_gold_data(ticker, days_back)

if df.empty:
    st.error(f"No se pudieron obtener datos para {ticker}. Intenta reducir los días a 30 o hacer clic en Actualizar.")
    st.stop()

# ==========================================
# MOTOR DE BACKTEST Y TRACKING
# ==========================================
trades = []
in_pos = False
p_entry = 0
idx_entry = 0
sl_price = 0
entry_timestamps = []
exit_timestamps = []
exit_prices = []
trade_motives = []

for i in range(len(df)):
    if not in_pos and df['signal'].iloc[i] == 1:
        in_pos = True
        p_entry = df['close'].iloc[i]
        idx_entry = i
        sl_price = p_entry * (1 - sl_pct)
        entry_timestamps.append(df.index[i])
        continue

    if in_pos:
        p_close = df['close'].iloc[i]
        p_high = df['high'].iloc[i]
        p_low = df['low'].iloc[i]
        target_vwap = df['vwap'].iloc[i]
        bars = i - idx_entry

        # Take Profit al VWAP o +0.80%
        if p_high >= target_vwap or p_close >= p_entry * (1 + tp_min_pct):
            exit_p = max(target_vwap, p_entry * (1 + tp_min_pct))
            ret = (exit_p - p_entry) / p_entry
            trades.append(ret)
            exit_timestamps.append(df.index[i])
            exit_prices.append(exit_p)
            trade_motives.append("TP (VWAP Reversion)")
            in_pos = False
        # Stop Loss
        elif p_low <= sl_price:
            trades.append(-sl_pct)
            exit_timestamps.append(df.index[i])
            exit_prices.append(sl_price)
            trade_motives.append(f"SL (-{sl_pct*100:.2f}%)")
            in_pos = False
        # Timeout (100 min)
        elif bars >= 20:
            ret = (p_close - p_entry) / p_entry
            trades.append(ret)
            exit_timestamps.append(df.index[i])
            exit_prices.append(p_close)
            trade_motives.append("Timeout (100m)")
            in_pos = False

# ==========================================
# MÉTRICAS Y CABECERA
# ==========================================
st.title(f"🟡 {ticker} — Monitor de Reversión a Bandas VWAP")
st.caption(f"Estrategia Institucional en Velas de 5 Minutos | Ventana RTH | Muestra: {days_back} Días")

df_trades = pd.Series(trades) if trades else pd.Series(dtype=float)

col1, col2, col3, col4, col5 = st.columns(5)

if not df_trades.empty:
    wins = df_trades[df_trades > 0]
    losses = df_trades[df_trades < 0]
    wr = (len(wins) / len(df_trades)) * 100
    pf = abs(wins.sum() / (losses.sum() + 1e-6))
    tot_ret = df_trades.sum() * 100
    payoff = abs(wins.mean() / (losses.mean() + 1e-6)) if len(losses) > 0 else 999.0

    col1.metric("Trades Totales", len(df_trades))
    col2.metric("Win Rate", f"{wr:.1f}%")
    col3.metric("Profit Factor", f"{pf:.2f}")
    col4.metric("Payoff Ratio", f"{payoff:.2f}x")
    col5.metric("Retorno Acumulado", f"{tot_ret:+.2f}%")
else:
    col1.metric("Trades Totales", "0")
    col2.metric("Win Rate", "N/A")
    col3.metric("Profit Factor", "N/A")
    col4.metric("Payoff Ratio", "N/A")
    col5.metric("Retorno Acumulado", "0.00%")

st.markdown("---")

# ==========================================
# ESTADO DEL RADAR EN VIVO
# ==========================================
last_bar = df.iloc[-1]
is_active_signal = last_bar['signal'] == 1
current_price = last_bar['close']
current_vwap = last_bar['vwap']
current_z = last_bar['vwap_zscore']

if is_active_signal:
    st.success(f"🚨 **SEÑAL ACTIVA EN TIEMPO REAL:** Suelo de compresión en {ticker} a ${current_price:,.2f}. Objetivo VWAP: ${current_vwap:,.2f}")
else:
    st.info(f"📡 **Radar en Espera:** Precio: ${current_price:,.2f} | VWAP Sesión: ${current_vwap:,.2f} | Desviación Z-Score: {current_z:.2f}σ")

# ==========================================
# GRÁFICO INTERACTIVO PLOTLY
# ==========================================
df_plot = df.iloc[-250:].copy()

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=[0.75, 0.25],
    subplot_titles=(f"Microestructura de {ticker} con Banda VWAP", "Volumen y Absorción")
)

# Velas
fig.add_trace(go.Candlestick(
    x=df_plot.index, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'],
    name="Precio", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
), row=1, col=1)

# VWAP y Banda
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['vwap'], line=dict(color='#FFD700', width=1.5), name="VWAP Sesión"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['vwap_lower_10'], line=dict(color='#42A5F5', width=1, dash='dot'), name=f"Banda {z_threshold:.2f}σ"), row=1, col=1)

# Marcadores de Señal
signals_plot = df_plot[df_plot['signal'] == 1]
if not signals_plot.empty:
    fig.add_trace(go.Scatter(
        x=signals_plot.index, y=signals_plot['low'] * 0.998, mode='markers',
        marker=dict(symbol='triangle-up', size=13, color='#00FF66'),
        name="Señal Entrada (Long)"
    ), row=1, col=1)

# Volumen
fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'], marker_color='#78909C', name="Volumen"), row=2, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['vol_ma20'], line=dict(color='#FFCA28', width=1), name="Media Vol 20"), row=2, col=1)

fig.update_layout(
    height=600, xaxis_rangeslider_visible=False, template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# CURVA DE CAPITAL Y REGISTRO DE TRADES
# ==========================================
if not df_trades.empty:
    tab1, tab2 = st.tabs(["📈 Curva de Capital Acumulada", "📋 Registro de Operaciones"])
    
    with tab1:
        equity_curve = (1 + df_trades).cumprod() - 1
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            y=equity_curve.values * 100, mode='lines+markers',
            line=dict(color='#FFD700', width=2), name='Retorno Neto (%)'
        ))
        fig_equity.update_layout(
            title="Curva de Capital del Bot de Oro (GLD)",
            xaxis_title="# de Operación", yaxis_title="Retorno Acumulado (%)",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig_equity, use_container_width=True)

    with tab2:
        df_hist = pd.DataFrame({
            "Entrada": entry_timestamps[:len(df_trades)],
            "Salida": exit_timestamps[:len(df_trades)],
            "Motivo": trade_motives[:len(df_trades)],
            "Precio Salida": [f"${p:,.2f}" for p in exit_prices[:len(df_trades)]],
            "Retorno": [f"{r*100:+.2f}%" for r in df_trades]
        })
        st.dataframe(df_hist.iloc[::-1], use_container_width=True)
