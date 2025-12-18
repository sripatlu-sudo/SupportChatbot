from twilio.rest import Client
account_sid = 'AC59097ed255cf8369cefef48e1ead1d92'
auth_token = '299956a49d94fbb4c3c63a38e9dc49ac'
client = Client(account_sid, auth_token)
text_message = "Hello! This SMS was sent using Python and Twilio"
message = client.messages.create(
  from_='+18669015477',
  body='HELLO',
  to='+18777804236'
)

print(message.sid)