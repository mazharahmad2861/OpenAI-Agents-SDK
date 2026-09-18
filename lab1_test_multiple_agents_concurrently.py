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


research_agent = Agent(
    name="Researcher",
    instructions="Research the topic.",
    model=model
)

summary_agent = Agent(
    name="Summarizer",
    instructions="Create a concise summary.",
    model=model
)


async def main():

    research, summary = await asyncio.gather(

        Runner.run(
            research_agent,
            "Research Google A2A Protocol."
        ),

        Runner.run(
            summary_agent,
            "Summarize what an Google A2A is."
        )
    )

    print("Research:")
    print(research.final_output)

    print("\nSummary:")
    print(summary.final_output)


asyncio.run(main())