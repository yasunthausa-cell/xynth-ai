import os
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def scrape_website(url: str) -> str:
    """Scrapes and extracts text content from a given website URL. Useful for reading documentation, articles, or extracting data from a specific page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        text = ' '.join(soup.stripped_strings)
        return text[:6000] # Limit tokens to avoid overwhelming the context window
    except Exception as e:
        return f"Failed to scrape {url}: {str(e)}"

@tool
def execute_python_code(code: str) -> str:
    """Executes simple python code and returns the output. Use this for math, string manipulation, or basic data processing."""
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        # We use a restricted dictionary for safety, but it still executes locally
        exec(code, {"__builtins__": __builtins__}, {})
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully with no output."
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = old_stdout

@tool
def save_text_to_file(filename: str, content: str) -> str:
    """Saves text content to a local file. Useful for saving scraped data, reports, or automation results."""
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully saved content to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool
def call_api(url: str, method: str = "GET", payload: dict = None) -> str:
    """Makes an HTTP request to an API and returns the JSON response. Useful for connecting to external services."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"API Call failed: {str(e)}"

def main():
    print("="*60)
    print("🤖🚀 Xynth AI - The Superagent")
    print("Powered by Groq and LangGraph")
    print("="*60)

    if not os.environ.get("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY is not set. Please add it to your Secrets.")
        return

    # Tool-capable Groq-hosted model
    model_name = "llama-3.3-70b-versatile"
    try:
        llm = ChatGroq(model=model_name, temperature=0.1)
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}")
        return

    # Initialize tools
    search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description="Search the web for current information, news, or to find URLs to scrape."
    )

    tools = [search_tool, scrape_website, execute_python_code, save_text_to_file, call_api]

    # Create the ReAct agent
    # We use LangGraph's prebuilt ReAct agent which handles tool execution loops automatically
    system_prompt = SystemMessage(content="""You are Xynth AI, a powerful superagent Created By Aether Aiko. The creator is Yasuntha Ravihara.
You are capable of autonomous actions, data scraping, web searching, making API calls, and executing python workflows.
Always be helpful, precise, and use tools when necessary to fulfill the user's request. 
If you need to find information, use the web_search tool. 
If you need to read a specific page, use the scrape_website tool.
If you need to perform calculations or data processing, use the execute_python_code tool.
If you need to save data or output, use the save_text_to_file tool.
If you need to interact with external web services, use the call_api tool.""")

    agent = create_react_agent(llm, tools)

    while True:
        try:
            user_query = input("\n👤 You: ")
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Xynth AI powering down...")
                break

            print("🤖 Xynth is thinking......")

            # Streaming the chunks to show tool execution
            messages = [system_prompt, HumanMessage(content=user_query)]
            for chunk in agent.stream(
                {"messages": messages},
                stream_mode="values"
            ):
                message = chunk["messages"][-1]
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"   [🛠️ Tool Call] Using {tool_call['name']}...")

            # Final output is the last message content
            final_message = chunk["messages"][-1].content
            print(f"\n✨ Xynth AI: {final_message}")

        except KeyboardInterrupt:
            print("\n👋 Xynth AI powering down...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
