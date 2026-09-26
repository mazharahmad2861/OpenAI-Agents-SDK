import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import TavilyClient

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)


load_dotenv()

set_tracing_disabled(True)


# -------------------------
# Groq
# -------------------------

groq_client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)


model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=groq_client
)


# -------------------------
# Tavily
# -------------------------

tavily_client = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)


# -------------------------
# Search Tool
# -------------------------

@function_tool
def web_search(query: str) -> str:
    """Search the web for current information."""

    response = tavily_client.search(
        query=query,
        max_results=5
    )

    results = []

    for result in response["results"]:
        results.append(
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Content: {result['content']}"
        )

    return "\n\n".join(results)


# -------------------------
# Agent
# -------------------------

search_agent = Agent(
    name="Search Agent",

    instructions="""
    You are a web search agent.

    Use the web_search tool when the user asks for
    current or web-based information.

    Analyze the search results and provide a concise,
    accurate answer.

    Include relevant source URLs in your answer.
    Do not make up information.
    """,

    model=model,

    tools=[web_search]
)


# -------------------------
# Run Agent
# -------------------------

query = "What are the best agentic AI frameworks for building autonomous agents in 2026?"

async def main():

    result = await Runner.run(
        search_agent,
        query
    )

    print("\nFINAL ANSWER:\n")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())