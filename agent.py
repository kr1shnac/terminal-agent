import os
from openai import OpenAI 
from dotenv import  load_dotenv

load_dotenv ()

history = []

client = OpenAI (
    base_url="https://openrouter.ai/api/v1",
    api_key= os.environ.get("OPENROUTER_API_KEY")
)


while True : 
    try :
        user_input = input ("You > ")

        if not user_input:
            continue

        if user_input == '/exit' :
            print("bye bye")
            break

        history.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model = "poolside/laguna-s-2.1:free",
            messages= [
                {"role": "system", "content": "I am retian Always call me with a name called retain and I am a useful agent which will help the user to guide."}
            ] + history
        )

        reply = response.choices[0].message.content

        history.append({"role": "assistent", "content": reply })

        print("Retain > ", reply)

    except Exception as e:
        print("Error: ", e)
        history.pop() 