"""Interactive CLI for Xynth AI. For programmatic / WhatsApp access, run api.py instead."""
from agent import XynthRunner, close_browser


def main():
    print("=" * 60)
    print("🤖🚀 Xynth AI - The Superagent")
    print("=" * 60)

    try:
        runner = XynthRunner()
    except Exception as e:
        print(f"❌ {e}")
        return

    print(f"\n💡 Active model: {runner.current_model}  (will fall back if needed)\n")
    print("Commands: 'exit' quit · 'reset' clear memory · 'models' list models · 'use <n>' switch model\n")

    while True:
        try:
            user_query = input("👤 You: ").strip()
            if not user_query:
                continue
            low = user_query.lower()
            if low in ("exit", "quit"):
                print("👋 Xynth AI powering down…")
                break
            if low == "reset":
                runner.seen_sessions.clear()
                print("🧹 Memory cleared.\n")
                continue
            if low == "models":
                print("\n🧠 Available models:")
                for i, (name, _) in enumerate(runner.agents, 1):
                    marker = "👉" if i - 1 == runner.current_idx else "  "
                    print(f"   {marker} {i}. {name}")
                print("\nType 'use <number>' to switch (e.g. 'use 2').\n")
                continue
            if low.startswith("use "):
                arg = user_query[4:].strip()
                idx = None
                if arg.isdigit():
                    n = int(arg) - 1
                    if 0 <= n < len(runner.agents):
                        idx = n
                else:
                    for i, (name, _) in enumerate(runner.agents):
                        if arg.lower() in name.lower():
                            idx = i
                            break
                if idx is None:
                    print(f"⚠️  No model matching '{arg}'. Type 'models' to see the list.\n")
                else:
                    runner.current_idx = idx
                    print(f"✅ Switched to {runner.current_model}\n")
                continue

            print("🤖 Xynth is thinking…")
            reply = runner.run("cli-session", user_query)
            print(f"\n✨ Xynth AI: {reply}\n")

        except KeyboardInterrupt:
            print("\n👋 Xynth AI powering down…")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}\n")

    close_browser()


if __name__ == "__main__":
    main()
