import yfinance as yf
from datetime import date, timedelta

# Define a 60-day window
end_date = date.today()
start_date = end_date - timedelta(days=60)

stock = yf.Ticker("AAPL") # Example: use your ticker here
#data = ticker.history(start=start_date, end=end_date)
data_4h = stock.history(period="1mo", interval="4h")
print(data_4h)