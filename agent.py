import os
from dotenv import load_dotenv
from openai import OpenAI

import json
import subprocess #run_shell

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from rich.align import Align
from rich.panel import Panel

from memory import Memory
from router.router import route

console = Console()

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

history = []
memory = Memory()

#defining tool - telling AI about tool and what they do
TOOLS = [
    {
        "type": "function", #RUN SHELL
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run"
                    }
                },
                "required": ["command"]
            }
        } 
    }, {
        "type": "function",  #WRITE
        "function": {
            "name": "write_file",
            "description": "Write the file according to users input",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "understand the file path or location"
                    },
                    "content": {
                        "type": "string",
                        "description": "the AI return content will be extrated to write in file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    }, {
        "type": "function", #EDIT
        "function": {
            "name": "edit_file",
            "description": "Find a specific piece of text in a file and replace it with new text. Use this to make surgical edits without rewriting the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to edit."
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find and replace."
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The new text to replace it with."
                    }
                },
                "required": ["file_path", "old_text", "new_text"]
            }

        }
    }, {
        "type": "function", #READ
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file and return it as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to read. Example: main.py"
                    }
                },
                "required": ["file_path"]
            }
        }
    }, {
        "type": "function", #LIST
        "function": {
            "name": "list_directory",
            "description": "List all files and folder inside a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list. Use '.' for current directory."
                    }
                },
                "required": ["path"]
            }
        }
    }, {
        "type": "function", #CREATE_DIR
        "function": {
            "name": "create_directory",
            "description": "Create a new folder. Also creates any missing parent folders",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The folder path to create. Example: projects/my_app"
                    }
                },
                "required": ["path"]
            }
        }
    }, {
        "type": "function", #search in files
        "function": {
            "name": "search_in_files",
            "description": "Search for a word or a patter across all files in a directory. Returns every line that contain the match, with the file name and line number",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The word or text to serch for."
                    },
                    "directory": {
                        "type": "string",
                        "description": "The directory to search inside. Use '.' for current directory."
                    }
                },
                "required": ["pattern", "directory"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store an important fact, preference, goal, or event about the user for future conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The concise information that should be remembered."
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "The type of memory.",
                        "enum": ["fact", "preference", "goal", "event"]
                    }
                },
                "required": ["text", "memory_type"]
            }
        }
    },
]

#Tool EXECUTOR 

#COMMAND SHELL
def run_shell (command):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True #o/p as string not bytes
    )

    output = result.stdout #output

    if result.stderr: #error
        output = output + "\nError:\n" + result.stderr
    
    return output if output else "no output"



#WRITE
def write_file(file_path, content) :
    dir_name = os.path.dirname(file_path)

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(content))

    return f"success, file '{file_path}' written completely"



#EDIT
def edit_file(file_path, old_text, new_text):
    try: 
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if old_text not in content:
            return f"Error: could not find the text to replace in '{file_path}'"
        
        updated = content.replace(old_text, new_text, 1)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated)
        
        return f"Success: edit applied to '{file_path}'"

    except FileNotFoundError:
        return f"Error: file '{file_path}' not found"

    except Exception as e:
        return f"Error editing file: {str(e)}"



#READ
def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    except FileNotFoundError: 
        return f"Error: file '{file_path}' not found"

    except Exception as e:
        return f"Error reading file: {str(e)}"


#LIST_DIR
def list_directory(path):
    try: 
        items = os.listdir(path)
        
        if not items:
            return f"Directory '{path}' is empty"

        result = []

        for item in sorted(items):
            full = os.path.join(path, item)
            tag = "[DIR] " if os.path.isdir(full) else "[FILE]"
            result.append(f"{tag} {item}")
        
        return "\n".join(result)

    except FileNotFoundError:
        return f"Error: directory '{path}' not found"

    except Exception as e:
        return f"Error listing directory: {str(e)}"


#CREATE_DIR
def create_directory(path):
    try: 
        os.makedirs(path, exist_ok=True)
        return f"Success: directory '{path}' created"

    except Exception as e:
        return f"Error creating directory: {str(e)}"


#SERACH IN FILES - should read 
def search_in_files(pattern, directory):
    matches = []

    try:
        for root, dirs, files in os.walk(directory):

            # Skip hidden folders and __pycache__
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d != "__pycache__"
            ]

            for file in files:
                if file.endswith(
                    (".py", ".txt", ".md", ".json",
                     ".js", ".ts", ".html", ".css")
                ):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line_num, line in enumerate(f, 1):
                                if pattern.lower() in line.lower():
                                    matches.append(
                                        f"{file_path}:{line_num} → {line.rstrip()}"
                                    )
                    except:
                        continue

        if not matches:
            return f"No matches found for '{pattern}' in '{directory}'."

        return "\n".join(matches)

    except Exception as e:
        return f"Error searching files: {str(e)}"

#REMEMORY
def remember(text, memory_type):
    memory.add(text, memory_type)
    return f"Memory saved: {text}"


# TOOL DISPACHER 

def dispatch_tool(tool_name, args):
    if tool_name == "run_shell":
        return run_shell(args["command"])
    elif tool_name == "write_file":
        return write_file(args["file_path"], args["content"])
    elif tool_name == "read_file":
        return read_file(args["file_path"])
    elif tool_name == "edit_file":
        return edit_file(args["file_path"], args["old_text"], args["new_text"])
    elif tool_name == "list_directory":
        return list_directory(args["path"])
    elif tool_name == "create_directory":
        return create_directory(args["path"])
    elif tool_name == "search_in_files":
        return search_in_files(args["pattern"], args["directory"])
    elif tool_name == "remember":
        return remember(args["text"], args["memory_type"])
    else:
        return f"Error: unknown tool '{tool_name}'"

#-------------------------
#Welcome message
console.print(
    Align.center(
        r"""
 _____  _____ _____  _    ___ _   _ 
|  __ \| ____|_   _| / \  |_ _| \ | |
| |__) |  _|   | |  / _ \  | ||  \| |
|  _  /| |___  | | / ___ \ | || |\  |
|_| \_\|_____| |_|/_/   \_\___|_| \_|
        """,
        style="bold bright_cyan"
    )
)
console.print(
    Align.center("[dim]my first terminal AGENT[/dim]")
)
console.print()

while True:
    try:
        user_input = console.input("[bold green]You > [/bold green]").strip()

        if not user_input:
            continue;

        if user_input == "/quit":
            console.log("Good bye")
            break;
        
        relevant_memories = memory.search(user_input)

        memory_context = ""

        for mem in relevant_memories:
            memory_context += f"- {mem['text']}\n"

        history.append({ "role": "user", "content": user_input })

        system_prompt = (
            "You are Retain, a helpful AI coding agent. "
            "You can read files, write files, edit files, run shell commands, "
            "list directories, create folders, and search across files. "
            "You also have long-term memory. "
            "Use retrieved memories when they are relevant. "
            "Never invent personal information that is not present in the conversation "
            "or retrieved memories. "
            "When the user explicitly shares a useful fact, preference, goal, or event "
            "that would be useful in a future conversation, use the remember tool to store it. "
            "When there is no reliable information about something, say that you don't know. "
            "Always verify your work — after writing or editing a file, read it back. "
            "After running code, check the output for errors and fix them."
        )

        if memory_context:
            system_prompt += (
                "\n\n Relevant long term memories about the user: \n"
                + memory_context
                + "\n use  these  memories only when they are relevent"
            )

        # Route the user's query once before the API/tool loop
        model, tier, route_reason = route(user_input)

        console.print(
            f"[dim cyan]Router → {tier} | {route_reason}[/dim cyan]"
        )

        while True:
            try:

                response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": system_prompt
                                }
                            ] + history,
                            tools=TOOLS
                )

            except Exception as e:
                console.print(f"\n[red]API error: {e}[/red]")
                history.pop() # remove the failed user message
                break

            message = response.choices[0].message

            # did the AI want a tool or just reply
        
            if message.tool_calls:

                tool_calls_list = []

                for tc in message.tool_calls:
                    one_item = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }

                    tool_calls_list.append(one_item)

                history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls_list
                })

                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    console.print(
                        f"\n[dim yellow]⚙  {tool_name} → "
                        f"{list(args.values())[0] if args else ''}[/dim yellow]"
                    )

                    result = dispatch_tool(tool_name, args)

                    preview = result[:120] + "..." if len(result) > 120 else result
                    console.print(f"[dim green]   {preview}[/dim green]")

                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })

                    continue #take tool context goes back to input 

            else:
                reply = message.content

                history.append({
                    "role": "assistant",
                    "content": reply
                })

                console.print("\n[bold purple]Retain ›[/bold purple]")
                console.print(Markdown(reply))
                console.print()

                #exit inner loop
                break

    except Exception as e:
        console.print(f"\n[red]API error: {e}[/red]")
        history.pop()
        break
