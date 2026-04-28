"""Interactive CLI for Xynth AI. For programmatic / WhatsApp access, run api.py instead."""
from langchain_core.messages import HumanMessage
from agent import build_agent, close_browser


def main():
    print("=" * 60)
    print("🤖🚀 Xynth AI - The Superagent")
    print("Powered by Groq + LangGraph")
    print("=" * 60)

    try:
        agent, system_prompt = build_agent()
    except Exception as e:
        print(f"❌ {e}")
        return

    thread_config = {
        "configurable": {"thread_id": "cli-session"},
        "recursion_limit": 15,
    }

    first_turn = True
    while True:
        try:
            user_query = input("\n👤 You: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Xynth AI powering down...")
                break

            print("🤖 Xynth is thinking...")
            messages = [system_prompt, HumanMessage(content=user_query)] if first_turn else [HumanMessage(content=user_query)]
            first_turn = False

            final_chunk = None
            try:
                for chunk in agent.stream({"messages": messages}, config=thread_config, stream_mode="values"):
                    final_chunk = chunk
                    message = chunk["messages"][-1]
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            print(f"   [🛠️ ] Using {tool_call['name']}...")
            except Exception as stream_err:
                err_text = str(stream_err)
                if "tool_use_failed" in err_text or "GraphRecursionError" in err_text:
                    print("⚠️  Hit a snag, retrying with a tighter hint...")
                    retry_messages = [HumanMessage(content=user_query + "\n\n(Reminder: use the minimum number of tool calls. Pick ONE tool per need. Stop and answer once you have enough info.)")]
                    final_chunk = None
                    for chunk in agent.stream({"messages": retry_messages}, config=thread_config, stream_mode="values"):
                        final_chunk = chunk
                else:
                    raise

            if final_chunk is not None:
                final_message = final_chunk["messages"][-1].content
                print(f"\n✨ Xynth AI: {final_message}")

        except KeyboardInterrupt:
            print("\n👋 Xynth AI powering down...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

    close_browser()


if __name__ == "__main__":
    main()
