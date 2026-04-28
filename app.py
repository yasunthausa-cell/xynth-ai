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
    print("Type 'exit' to quit. Type 'reset' to clear conversation memory.\n")

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
