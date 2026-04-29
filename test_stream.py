import sys
sys.path.append('C:/Users/whuzf/Downloads/NonstopAnotherQuarks/NonstopAnotherQuarks')
import agent
from langchain_core.messages import HumanMessage

a, sp = agent.build_agent()
msgs = [sp, HumanMessage(content="Hello!")]
config = {"configurable": {"thread_id": "test_thread"}, "recursion_limit": 50}

print("Testing stream...")
for update in a.stream({"messages": msgs}, config=config, stream_mode=["messages", "updates"]):
    if isinstance(update, tuple):
        kind, payload = update
        if kind == "messages":
            # Just print the type of payload to understand it
            print(f"MESSAGES payload type: {type(payload)}")
            if isinstance(payload, tuple):
                print(f"MESSAGES payload tuple length: {len(payload)}, types: {[type(x) for x in payload]}")
                msg = payload[0]
                print(f"Chunk class: {msg.__class__.__name__}, content: {msg.content}")
            elif isinstance(payload, list):
                print(f"MESSAGES payload list length: {len(payload)}, types: {[type(x) for x in payload]}")
                msg = payload[0]
                print(f"Chunk class: {msg.__class__.__name__}, content: {msg.content}")
        elif kind == "updates":
            print(f"UPDATES payload keys: {payload.keys()}")
    else:
        print(f"Unknown update type: {type(update)}, value: {update}")
        
