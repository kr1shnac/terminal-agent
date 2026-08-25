import os
from openai import OpenAI 
from dotenv import  load_dotenv

load_dotenv ()

history = []

client = OpenAI (
    base_url="https://openrouter.ai/api/v1",
    api_key= os.environ.get("OPENROUTER_API_KEY")
)


TOOLS = [
    {
    "type": "function",
    "function" : {
        "name" : "write_input",
        "description" : "write the file ",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "file_path": {
                    "type": "string",
                    "description" : "understand work"
                }
                "content" : {
                    "type": "string",
                    "description" : "create the work"
                }
            },
        },
        "required" : ["file_path","content"]
        },
    }   
    {
        "type" : "function",
        "function" :{
            "name" : "edit_file",
            "description"  : "edititng the file"
            "parameters" : {
                "type" : "object",
                "properites" : {
                    "type" : "string",
                    "description": "edit the file"
                } 
                "old_text" :{
                    "type" : "string",
                    "description" : "this is old text to before the editing"
                }
                "new_text" :{
                    "type" : "string",
                    "description": "this is the new text after the edit competion"
                },                   
            },
            "required" : ["edit_file","old_text","new_text"]
        }
]


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

    TOOLS_calls =[]

    for tc in message.tool_calls  (
            "one_item" : {
               "id" : "id.tool.call"
               "type" : "function",
               "function" : {
                   "name" : "tc.message",
                   "arugment" : "tc.arugment"
               } 
            }
        )

        reply = response[0].chat.content

        history.append = ( 
            {
            "id" : "tc.id",
            "name" : "tc.name",
            "arugment" : "tc.arugment"
            }
        )

        for tc in reply.tool_calls (
            "tools_name" : "tc.name",
            "args" : "json.loads(tc.function.args)"

            results = dispatch_tools("tool_name","args")

            preview = result
            console.print(result)

            history.append ({
                "name" : "tc.name",
                "args" : "tc.args",
                "content" : result
            })
        )
      

        reply = response.choices[0].message.content

        history.append({"role": "assistent", "content": reply })

        print("Retain > ", reply)

    except Exception as e:
        print("Error: ", e)
        history.pop() 