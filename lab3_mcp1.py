import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.mcp import MCPServerStreamableHttp


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# 2. Connect to Groq
# --------------------------------------------------

groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# --------------------------------------------------
# 3. Configure Groq model
# --------------------------------------------------

model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=groq_client
)


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

async def main():

    # Connect to MCP server
    params = {
        "url": "https://mcp.context7.com/mcp",
        "timeout": 60,
    }

    async with MCPServerStreamableHttp(
        name="Context7",
        params=params,
    ) as server:

        # --------------------------------------------------
        # 5. Create Agent
        # --------------------------------------------------

        agent = Agent(
            name="MCP Agent",

            instructions="""
            You are a technical research assistant.

            Use the available MCP tools when useful.
            Use the MCP documentation to answer the question.
            Keep your answer simple and accurate.
            If the information is unavailable, say so.
            """,

            model=model,

            # Give MCP tools to the agent
            mcp_servers=[server],
        )


        # --------------------------------------------------
        # 6. Give the agent a task
        # --------------------------------------------------

        task = """
        Find information about the OpenAI Agents SDK.

        Explain:
        1. What is an Agent?
        2. What is Runner?
        3. What is MCP?

        Keep the explanation short and beginner-friendly.
        """


        # --------------------------------------------------
        # 7. Run the agent
        # --------------------------------------------------

        result = await Runner.run(
            agent,
            task,
        )


        # --------------------------------------------------
        # 8. Display result
        # --------------------------------------------------

        print("\n========== FINAL ANSWER ==========\n")
        print(result.final_output)


# --------------------------------------------------
# 9. Start
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())