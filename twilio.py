from twilio.rest import Client
account_sid = 'AC59097ed255cf8369cefef48e1ead1d92'
auth_token = '299956a49d94fbb4c3c63a38e9dc49ac'
client = Client(account_sid, auth_token)
message = client.messages.create(
    to='[13145048971]'
)
print(message.sid)