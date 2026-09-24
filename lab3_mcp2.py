import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel


load_dotenv()

# -----------------------------
# Groq client
# -----------------------------

groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=groq_client
)


async def main():

    # -----------------------------
    # Connect to MCP filesystem
    # -----------------------------

    async with MCPServerStdio(
        name="Filesystem",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "."
            ],
        },
    ) as server:

        # -----------------------------
        # Create Agent
        # -----------------------------

        agent = Agent(
            name="File Agent",
            instructions="""
            You are a file assistant.

            Use the MCP filesystem tools to inspect files.
            Do not guess information about files.
            """,
            model=model,
            mcp_servers=[server],
        )

        # -----------------------------
        # Run Agent
        # -----------------------------

        result = await Runner.run(
            agent,
            "List the files in the current directory."
        )

        print("\nAgent Response:")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())