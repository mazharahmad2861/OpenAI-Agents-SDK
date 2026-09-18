import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel

load_dotenv()


# Groq OpenAI-compatible client
groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


# Groq model
model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=groq_client,
)


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model,
)


async def main():

    result = await Runner.run(
        agent,
        "Explain what an AI agent is."
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())