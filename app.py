"""Interactive CLI for Resynth AI. For programmatic / WhatsApp access, run api.py instead."""
import sys
import threading
import itertools
import time
from agent import XynthRunner, close_browser


class Spinner:
    """Tiny animated braille spinner shown while the agent thinks."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "🤖 Resynth is thinking"):
        self.message = message
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        cycle = itertools.cycle(self.FRAMES)
        start = time.time()
        while not self._stop.is_set():
            elapsed = time.time() - start
            sys.stdout.write(f"\r{next(cycle)} {self.message}… ({elapsed:0.1f}s) ")
            sys.stdout.flush()
            time.sleep(0.08)
        # Clear the line on exit
        sys.stdout.write("\r" + " " * (len(self.message) + 24) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()


def main():
    print("=" * 60)
    print("🤖🚀 Resynth AI - The Superagent")
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
                print("👋 Resynth AI powering down…")
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

            with Spinner(f"🤖 Resynth ({runner.current_model}) is thinking"):
                reply = runner.run("cli-session", user_query)
            print(f"✨ Resynth AI: {reply}\n")

        except KeyboardInterrupt:
            print("\n👋 Resynth AI powering down…")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}\n")

    close_browser()


if __name__ == "__main__":
    main()
