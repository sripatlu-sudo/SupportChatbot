#!/usr/bin/env python3
"""
Stock Alert Daemon - Background service for SWING trading alerts

Run: python stock_alert_daemon.py
"""
import pandas as pd
import time
import json
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from plyer import notification
import yfinance as yf

# Configuration
REFRESH_INTERVAL = 60  # seconds
ALERT_LOG_FILE = "trading_alerts.json"
TICKERS_FILE = "swing_trade_tickers.txt"
RECEPIENTS_FILE = "alert_recepients.txt"

# Email settings (optional)
EMAIL_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = os.getenv("OPENAI_TE")
EMAIL_PASS = os.getenv("OPENAI_TP")
ALERT_EMAIL = os.getenv("OPENAI_TAE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data_4h = stock.history(period="2mo", interval="4h")
        data_1d = stock.history(period="3mo", interval="1d")
        data_1h = stock.history(period="1mo", interval="1h")
        print(symbol)
        print(data_4h)
        print("----------------------------------------------------------------------------")
        print(data_1d)
        print("----------------------------------------------------------------------------")
        print(data_1h)
        print("----------------------------------------------------------------------------")
        return data_4h, data_1d, data_1h
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None, None, None

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
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    bb_mid = close.rolling(bb_period).mean()
    bb_std = close.rolling(bb_period).std()
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    
    kc_mid = close.rolling(kc_period).mean()
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(kc_period).mean()
    kc_upper = kc_mid + (1.5 * atr)
    kc_lower = kc_mid - (1.5 * atr)
    
    squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
    return "yellow" if squeeze else "red"

def check_bb_breakdown(data_1h, period=20):
    close = data_1h['Close']
    bb_mid = close.rolling(period).mean()
    bb_std = close.rolling(period).std()
    bb_lower = bb_mid - (2 * bb_std)
    
    current_price = close.iloc[-1]
    prev_price = close.iloc[-2]
    
    breakdown = (current_price < bb_lower.iloc[-1]) and (prev_price >= bb_lower.iloc[-2])
    return breakdown

def analyze_stock(symbol, data_4h, data_1d, data_1h):
    if data_4h is None or data_1d is None or data_1h is None or len(data_4h) < 50 or len(data_1d) < 50 or len(data_1h) < 50:
        return "HOLD", "Insufficient data"
    
    print(f"Analyzing stock: {symbol}")

    current_price = data_1d['Close'].iloc[-1]
    
    rsi_4h = calculate_rsi(data_4h['Close'])
    rsi_1d = calculate_rsi(data_1d['Close'])
    sma_21 = data_1d['Close'].rolling(21).mean().iloc[-1]
    macd_line, macd_signal, macd_hist = calculate_macd(data_1d['Close'])
    ttm_squeeze = calculate_ttm_squeeze(data_1d)
    bb_breakdown = check_bb_breakdown(data_1h)
    print(f"Details of analysis: {rsi_4h},{rsi_1d},{sma_21},{macd_line},{macd_signal},{ttm_squeeze},{bb_breakdown}")
    # BUY Rules
    buy_conditions = [
        rsi_4h > 55,
        current_price > sma_21,
        rsi_1d > 50,
        ttm_squeeze == "yellow",
        macd_line > macd_signal and macd_hist > 0,
        current_price >= sma_21
    ]
    
    # SELL Rules
    sell_conditions = [
        rsi_1d < 50,
        current_price < sma_21,
        macd_line < macd_signal,
        bb_breakdown
    ]
    
    if all(buy_conditions):
        print("BUY signal conditions met...")
        return "BUY", f"All BUY conditions met: RSI_4H={rsi_4h:.1f}, RSI_1D={rsi_1d:.1f}, Price=${current_price:.2f}"
    elif all(sell_conditions):
        print("SELL signal conditions met...")
        return "SELL", f"🔥 GTFO SELL: RSI_1D={rsi_1d:.1f}, Price=${current_price:.2f}, BB breakdown"
    else:
        print("HOLD signal conditions met...")
        return "HOLD", f"Conditions not met: RSI_1D={rsi_1d:.1f}, Price=${current_price:.2f}"

def send_desktop_notification(title, message):
    try:
        notification.notify(
            title=title,
            message=message,
            timeout=10
        )
    except:
        pass

def load_email_recipients():
    try:
        with open(RECEPIENTS_FILE, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except:
        return [ALERT_EMAIL] if ALERT_EMAIL else []

def send_daemon_alert(subject, message):
    if not EMAIL_ENABLED or not all([EMAIL_USER, EMAIL_PASS]):
        return
      
    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_USER
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_USER, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Error processing email: {e}")

def send_email_alert(subject, message):
    if not EMAIL_ENABLED or not all([EMAIL_USER, EMAIL_PASS]):
        return
    
    recipients = load_email_recipients()
    if not recipients:
        return
    
    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = ', '.join(recipients)
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, recipients, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Error processing email: {e}")

def log_alert(alert):
    try:
        alerts = []
        if os.path.exists(ALERT_LOG_FILE):
            with open(ALERT_LOG_FILE, 'r') as f:
                alerts = json.load(f)
        
        alerts.append(alert)
        alerts = alerts[-100:]  # Keep last 100 alerts
        
        with open(ALERT_LOG_FILE, 'w') as f:
            json.dump(alerts, f, indent=2)
    except:
        pass

def load_tickers():
    try:
        with open(TICKERS_FILE, 'r') as f:
            return [line.strip() for line in f if line.strip()]
            print(line)
    except:
        return ["AAPL", "GOOGL", "MSFT", "NVDA", "MU", "ORCL", "CHTR"]

def main():
    symbols = load_tickers()
    print("🚀 Stock Alert Daemon Started")
    print(f"Monitoring: {', '.join(symbols)}")
    print(f"Refresh interval: {REFRESH_INTERVAL} seconds")
    print("Press Ctrl+C to stop\n")
    send_daemon_alert("Job alert","Successfully started daemon!")
    
    last_alerts = {}
    
    try:
        while True:
            for symbol in symbols:
                try:
                    data_4h, data_1d, data_1h = get_stock_data(symbol)
                    
                    if data_1d is not None and len(data_1d) > 0:
                        current_price = data_1d['Close'].iloc[-1]
                        signal, reason = analyze_stock(symbol, data_4h, data_1d, data_1h)
                        
                        if signal in ["BUY", "SELL"]:
                            # Check if this is a new alert
                            alert_key = f"{symbol}_{signal}"
                            if alert_key not in last_alerts or last_alerts[alert_key] != current_price:
                                
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                alert_title = f"🚨 {signal} {symbol}"
                                alert_message = f"Price: ${current_price:.2f}\n{reason}"
                                
                                # Log alert
                                alert_data = {
                                    'timestamp': timestamp,
                                    'symbol': symbol,
                                    'signal': signal,
                                    'price': current_price,
                                    'reason': reason
                                }
                                log_alert(alert_data)
                                
                                # Send notifications
                                # send_desktop_notification(alert_title, alert_message)
                                send_email_alert(alert_title, alert_message)
                                
                                # Console output
                                print(f"[{timestamp}] {alert_title} - ${current_price:.2f}")
                                print(f"  Reason: {reason}\n")
                                
                                last_alerts[alert_key] = current_price
                        
                        else:
                            # Remove from last_alerts if no longer signaling
                            for key in list(last_alerts.keys()):
                                if key.startswith(f"{symbol}_"):
                                    del last_alerts[key]
                
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
            
            time.sleep(REFRESH_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n🛑 Stock Alert Daemon Stopped")

if __name__ == "__main__":
    main()