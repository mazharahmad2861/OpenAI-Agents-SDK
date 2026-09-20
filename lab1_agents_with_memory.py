import asyncio
import os
import requests

from dotenv import load_dotenv
from agents import Agent, Runner, SQLiteSession, function_tool, trace


load_dotenv()


# ============================================================
# TOOL
# ============================================================

@function_tool
def send_push_notification(message: str) -> str:
    """Send a push notification to the user."""

    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")

    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": token,
            "user": user,
            "message": message,
        },
    )

    response.raise_for_status()

    return f"Push notification sent. Status code: {response.status_code}"


# ============================================================
# AGENT
# ============================================================

agent = Agent(
    name="Assistant",
    instructions=(
        "You are a helpful assistant. "
        "You can send push notifications when the user asks you to."
    ),
    model="gpt-5.4-mini",
    tools=[send_push_notification],
)


# ============================================================
# SESSION / MEMORY
# ============================================================

session = SQLiteSession("conversation_1", "memory.db")


# ============================================================
# MAIN
# ============================================================

async def main():

    # --------------------------------------------------------
    # 1. Normal Runner.run()
    # --------------------------------------------------------

    with trace("Assistant Conversation"):

        result = await Runner.run(
            agent,
            "Hi, my name is Mazhar.",
            session=session,
        )

        print("\nAssistant:")
        print(result.final_output)


        # ----------------------------------------------------
        # 2. Memory
        # ----------------------------------------------------

        result = await Runner.run(
            agent,
            "What's my name?",
            session=session,
        )

        print("\nAssistant:")
        print(result.final_output)


        # ----------------------------------------------------
        # 3. Tool calling
        # ----------------------------------------------------

        result = await Runner.run(
            agent,
            "Send me a notification saying: The pizza is here!",
            session=session,
        )

        print("\nAssistant:")
        print(result.final_output)


    # --------------------------------------------------------
    # 4. Streaming
    # --------------------------------------------------------

    print("\nStreaming response:\n")

    streamed_result = Runner.run_streamed(
        agent,
        "Explain what an A2A protocol is in a few sentences.",
    )

    async for event in streamed_result.stream_events():
        print(event)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())