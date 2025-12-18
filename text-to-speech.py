from openai import OpenAI
client = OpenAI()

response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input=open("script.txt").read()
)

with open("cathy_john_high_quality.mp3", "wb") as f:
    f.write(response)