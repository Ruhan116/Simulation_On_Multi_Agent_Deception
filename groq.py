import requests
import json

api_key = "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"

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

conversation = []

print("Welcome to Groq Chat! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    conversation.append({"role": "user", "content": user_input})

    data = {
        "model": "llama3-70b-8192",  #suppported model
        "messages": conversation,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]
        print("Groq:", assistant_message)
        conversation.append({"role": "assistant", "content": assistant_message})
    else:
        print("Error:", response.status_code)
        try:
            print(response.json())
        except:
            print(response.text)
