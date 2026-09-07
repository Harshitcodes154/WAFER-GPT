from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="You are WAFER GPT. Explain what an Edge-Ring wafer defect means in 3 simple points."
)

print(response.text)