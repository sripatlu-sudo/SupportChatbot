import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
from openai import OpenAI
import os

st.set_page_config(page_title="Stock Trader AI", layout="wide")

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #00C851;
    font-size: 36px;
    font-weight: bold;
    margin-bottom: 20px;
}
.alert-box {
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    font-weight: bold;
}
.buy-alert {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    color: #155724;
}
.sell-alert {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
    color: #721c24;
}
.hold-alert {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 Stock Trader AI Agent</div>', unsafe_allow_html=True)

# Initialize session state
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="5d", interval="1m")
        info = stock.info
        return data, info
    except:
        return None, None

def analyze_stock(symbol, data, info):
    if data is None or len(data) < 20:
        return "HOLD", "Insufficient data"
    
    current_price = data['Close'].iloc[-1]
    sma_20 = data['Close'].rolling(20).mean().iloc[-1]
    rsi = calculate_rsi(data['Close'])
    
    # Simple trading logic
    if current_price > sma_20 and rsi < 30:
        return "BUY", f"Price above SMA20 (${sma_20:.2f}) and RSI oversold ({rsi:.1f})"
    elif current_price < sma_20 and rsi > 70:
        return "SELL", f"Price below SMA20 (${sma_20:.2f}) and RSI overbought ({rsi:.1f})"
    else:
        return "HOLD", f"Price: ${current_price:.2f}, SMA20: ${sma_20:.2f}, RSI: {rsi:.1f}"

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def get_ai_analysis(symbol, current_price, signal, reason):
    try:
        client = get_openai_client()
        prompt = f"""
        Analyze {symbol} stock:
        Current Price: ${current_price:.2f}
        Technical Signal: {signal}
        Reason: {reason}
        
        Provide a brief trading recommendation (2-3 sentences max).
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content
    except:
        return "AI analysis unavailable"

# Sidebar controls
with st.sidebar:
    st.subheader("🎯 Trading Setup")
    
    symbols = st.text_input("Stock Symbols (comma-separated)", "AAPL,GOOGL,MSFT").upper().split(',')
    symbols = [s.strip() for s in symbols if s.strip()]
    
    refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60)
    
    if st.button("🚀 Start Monitoring"):
        st.session_state.monitoring = True
        st.success("Monitoring started!")
    
    if st.button("⏹️ Stop Monitoring"):
        st.session_state.monitoring = False
        st.info("Monitoring stopped!")
    
    if st.button("🗑️ Clear Alerts"):
        st.session_state.alerts = []

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Live Stock Data")
    
    if symbols:
        for symbol in symbols:
            data, info = get_stock_data(symbol)
            
            if data is not None and len(data) > 0:
                current_price = data['Close'].iloc[-1]
                change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
                change_pct = (change / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
                
                # Display stock info
                st.metric(
                    label=f"{symbol}",
                    value=f"${current_price:.2f}",
                    delta=f"{change_pct:+.2f}%"
                )
                
                # Create chart
                fig = go.Figure(data=go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close']
                ))
                fig.update_layout(title=f"{symbol} Price Chart", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyze and generate alerts
                if st.session_state.monitoring:
                    signal, reason = analyze_stock(symbol, data, info)
                    
                    if signal in ["BUY", "SELL"]:
                        ai_analysis = get_ai_analysis(symbol, current_price, signal, reason)
                        
                        alert = {
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'symbol': symbol,
                            'signal': signal,
                            'price': current_price,
                            'reason': reason,
                            'ai_analysis': ai_analysis
                        }
                        
                        # Add to alerts if not duplicate
                        if not any(a['symbol'] == symbol and a['signal'] == signal for a in st.session_state.alerts[-5:]):
                            st.session_state.alerts.append(alert)

with col2:
    st.subheader("🚨 Trading Alerts")
    
    if st.session_state.alerts:
        for alert in reversed(st.session_state.alerts[-10:]):  # Show last 10 alerts
            alert_class = "buy-alert" if alert['signal'] == "BUY" else "sell-alert"
            
            st.markdown(f"""
            <div class="alert-box {alert_class}">
                <strong>{alert['signal']} {alert['symbol']}</strong><br>
                Price: ${alert['price']:.2f}<br>
                Time: {alert['time']}<br>
                Reason: {alert['reason']}<br>
                AI: {alert['ai_analysis']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No alerts yet. Start monitoring to see trading signals!")

# Auto-refresh when monitoring
if st.session_state.monitoring:
    time.sleep(refresh_interval)
    st.rerun()

# Status indicator
status_color = "🟢" if st.session_state.monitoring else "🔴"
st.sidebar.markdown(f"**Status:** {status_color} {'Monitoring' if st.session_state.monitoring else 'Stopped'}")