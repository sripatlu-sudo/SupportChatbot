import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
from openai import OpenAI
import os
# from twilio.rest import Client

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
.ticker-blue {
    background-color: #e3f2fd;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.ticker-green {
    background-color: #e8f5e8;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 Stock Trader AI Agent</div>', unsafe_allow_html=True)

# Initialize session state
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = True  # Start monitoring by default

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data_4h = stock.history(period="60d", interval="4h")
        data_1d = stock.history(period="100d", interval="1d")
        data_1h = stock.history(period="30d", interval="1h")
        info = stock.info
        return data_4h, data_1d, data_1h, info
    except:
        return None, None, None, None

def analyze_stock(symbol, data_4h, data_1d, data_1h, info):
    if data_4h is None or data_1d is None or data_1h is None or len(data_4h) < 50 or len(data_1d) < 50 or len(data_1h) < 50:
        return "HOLD", "Insufficient data"
    
    current_price = data_1d['Close'].iloc[-1]
    
    # Calculate indicators
    rsi_4h = calculate_rsi(data_4h['Close'])
    rsi_1d = calculate_rsi(data_1d['Close'])
    sma_21 = data_1d['Close'].rolling(21).mean().iloc[-1]
    macd_line, macd_signal, macd_hist = calculate_macd(data_1d['Close'])
    ttm_squeeze = calculate_ttm_squeeze(data_1d)
    bb_breakdown = check_bb_breakdown(data_1h)
    
    # BUY Rules
    buy_conditions = [
        rsi_4h > 55,  # RSI > 55 on 4H
        current_price > sma_21,  # Price above 21-day MA
        rsi_1d > 50,  # RSI > 50 on 1D
        ttm_squeeze == "yellow",  # TTM Squeeze Yellow
        macd_line > macd_signal and macd_hist > 0,  # MACD bullish
        current_price >= sma_21  # Price at/above 21-day MA
    ]
    
    # NEW SELL Rules
    sell_conditions = [
        rsi_1d < 50,  # RSI < 50 (Weak Momentum)
        current_price < sma_21,  # Price below 21-day MA (No uptrend)
        macd_line < macd_signal,  # MACD bearish crossover
        bb_breakdown  # GTFO Rule - 1H Bollinger Band Breakdown
    ]
    
    if all(buy_conditions):
        return "BUY", f"All BUY conditions met: RSI_4H={rsi_4h:.1f}, RSI_1D={rsi_1d:.1f}, Price=${current_price:.2f}, MA21=${sma_21:.2f}"
    elif all(sell_conditions):
        return "SELL", f"🔥 GTFO SELL: RSI_1D={rsi_1d:.1f} (<50), Price=${current_price:.2f} below MA21=${sma_21:.2f}, MACD bearish, BB breakdown"
    else:
        return "HOLD", f"Conditions not met: RSI_1D={rsi_1d:.1f}, Price=${current_price:.2f}, MA21=${sma_21:.2f}"

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_hist = macd_line - macd_signal
    return macd_line.iloc[-1], macd_signal.iloc[-1], macd_hist.iloc[-1]

def calculate_ttm_squeeze(data, bb_period=20, kc_period=20):
    # Simplified TTM Squeeze indicator
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # Bollinger Bands
    bb_mid = close.rolling(bb_period).mean()
    bb_std = close.rolling(bb_period).std()
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    
    # Keltner Channels (simplified)
    kc_mid = close.rolling(kc_period).mean()
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(kc_period).mean()
    kc_upper = kc_mid + (1.5 * atr)
    kc_lower = kc_mid - (1.5 * atr)
    
    # Squeeze condition
    squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
    
    if squeeze:
        return "yellow"  # Squeeze on
    else:
        return "red"  # Squeeze off

def check_bb_breakdown(data_1h, period=20):
    # GTFO Rule - 1-Hour Bollinger Band Breakdown
    close = data_1h['Close']
    bb_mid = close.rolling(period).mean()
    bb_std = close.rolling(period).std()
    bb_lower = bb_mid - (2 * bb_std)
    
    current_price = close.iloc[-1]
    prev_price = close.iloc[-2]
    
    # Check if price broke below lower Bollinger Band
    breakdown = (current_price < bb_lower.iloc[-1]) and (prev_price >= bb_lower.iloc[-2])
    
    return breakdown

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

# def send_sms_alert(phone_number, message):
#     try:
#         # Twilio credentials (set as environment variables)
#         account_sid = os.getenv('TWILIO_ACCOUNT_SID')
#         auth_token = os.getenv('TWILIO_AUTH_TOKEN')
#         from_number = os.getenv('TWILIO_PHONE_NUMBER')
#         
#         if account_sid and auth_token and from_number:
#             client = Client(account_sid, auth_token)
#             client.messages.create(
#                 body=message,
#                 from_=from_number,
#                 to=phone_number
#             )
#             return True
#     except:
#         pass
#     return False

# Sidebar controls
with st.sidebar:
    st.subheader("🎯 Trading Setup")
    
    symbols = st.text_input("Stock Symbols (comma-separated)", "AAPL,GOOGL,MSFT, NVDA, MU, ORCL, CHTR").upper().split(',')
    symbols = [s.strip() for s in symbols if s.strip()]
    
    # phone_number = st.text_input("📱 Phone Number (for SMS alerts)", placeholder="+1234567890")
    
    refresh_interval = st.slider("Refresh Interval (seconds)", 10, 28800, 10)  # Default 10 seconds
    
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
        for idx, symbol in enumerate(symbols):
            data_4h, data_1d, data_1h, info = get_stock_data(symbol)
            
            if data_1d is not None and len(data_1d) > 0:
                current_price = data_1d['Close'].iloc[-1]
                change = data_1d['Close'].iloc[-1] - data_1d['Close'].iloc[-2] if len(data_1d) > 1 else 0
                change_pct = (change / data_1d['Close'].iloc[-2] * 100) if len(data_1d) > 1 else 0
                
                # Calculate RSI for display
                rsi_4h = calculate_rsi(data_4h['Close']) if data_4h is not None and len(data_4h) > 14 else 0
                rsi_1d = calculate_rsi(data_1d['Close']) if len(data_1d) > 14 else 0
                
                # Alternate colors for each ticker
                color_class = "ticker-blue" if idx % 2 == 0 else "ticker-green"
                
                st.markdown(f'<div class="{color_class}">', unsafe_allow_html=True)
                
                # Display stock info with RSI
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    st.metric(
                        label=f"{symbol}",
                        value=f"${current_price:.2f}",
                        delta=f"{change_pct:+.2f}%"
                    )
                with col_b:
                    st.metric("RSI 4H", f"{rsi_4h:.1f}")
                with col_c:
                    st.metric("RSI 1D", f"{rsi_1d:.1f}")
                
                # Create chart
                fig = go.Figure(data=go.Candlestick(
                    x=data_1d.index,
                    open=data_1d['Open'],
                    high=data_1d['High'],
                    low=data_1d['Low'],
                    close=data_1d['Close']
                ))
                # Add 21-day MA
                fig.add_trace(go.Scatter(
                    x=data_1d.index,
                    y=data_1d['Close'].rolling(21).mean(),
                    name='21-day MA',
                    line=dict(color='blue')
                ))
                fig.update_layout(title=f"{symbol} Price Chart", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyze and generate alerts
                if st.session_state.monitoring:
                    signal, reason = analyze_stock(symbol, data_4h, data_1d, data_1h, info)
                    
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
                            
                            # Send SMS alert if phone number provided
                            # if phone_number:
                            #     sms_message = f"🚨 {signal} {symbol} at ${current_price:.2f}\n{reason[:100]}"
                            #     send_sms_alert(phone_number, sms_message)
                
                st.markdown('</div>', unsafe_allow_html=True)

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

# Auto-refresh when monitoring (persistent)
if st.session_state.monitoring:
    time.sleep(10)  # Fixed 10-second refresh for price and rules
    st.rerun()

# Status indicator
status_color = "🟢" if st.session_state.monitoring else "🔴"
st.sidebar.markdown(f"**Status:** {status_color} {'Monitoring' if st.session_state.monitoring else 'Stopped'}")