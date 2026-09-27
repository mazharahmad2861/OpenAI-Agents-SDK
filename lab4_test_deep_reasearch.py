import os
import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    function_tool,
    trace,
    OpenAIChatCompletionsModel,
)

from agents.model_settings import ModelSettings

from tavily import AsyncTavilyClient

from messenger import send_email, push


# ============================================================
# Configuration
# ============================================================

load_dotenv(override=True)

MODEL_NAME = "openai/gpt-oss-20b"

HOW_MANY_SEARCHES = 5
USE_EMAIL = True


# ============================================================
# Clients
# ============================================================

# Groq OpenAI-compatible client
groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Tavily client
tavily_client = AsyncTavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# Groq model for OpenAI Agents SDK
model = OpenAIChatCompletionsModel(
    model=MODEL_NAME,
    openai_client=groq_client,
)


# ============================================================
# Tool: Tavily Web Search
# ============================================================

@function_tool
async def tavily_search(query: str) -> str:
    """
    Search the web using Tavily and return relevant results.

    Args:
        query: The web search query.
    """

    response = await tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

    results = response.get("results", [])

    if not results:
        return "No search results found."

    formatted_results = []

    for result in results:
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")

        formatted_results.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(formatted_results)


# ============================================================
# Agent 1: Search Agent
# ============================================================

SEARCH_INSTRUCTIONS = """
You are a research assistant.

Given a search term, use the Tavily search tool to search
the web for current and relevant information.

After searching, produce a concise summary of the results.

The summary must be 2-3 paragraphs and less than 300 words.

Capture the main points and be succinct.

Reply only with the summary.
"""

search_agent = Agent(
    name="Search Agent",
    instructions=SEARCH_INSTRUCTIONS,
    tools=[tavily_search],
    model=model,
    model_settings=ModelSettings(
        tool_choice="required"
    ),
)


# ============================================================
# Agent 2: Planner Agent
# ============================================================

class WebSearchItem(BaseModel):
    reason: str = Field(
        description=(
            "Why this search is important for answering "
            "the user's question."
        )
    )

    query: str = Field(
        description="The search query to use."
    )


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(
        description=(
            "A list of web searches needed to answer "
            "the user's question."
        )
    )


PLANNER_INSTRUCTIONS = f"""
You are a research planning assistant.

Given a user query, create a research plan.

Identify the most useful web searches needed to answer
the question accurately.

Output exactly {HOW_MANY_SEARCHES} search queries.

For every search, first explain why the search is important,
then provide the actual search query.
"""

planner_agent = Agent(
    name="Planner Agent",
    instructions=PLANNER_INSTRUCTIONS,
    model=model,
    output_type=WebSearchPlan,
)


# ============================================================
# Agent 3: Writer Agent
# ============================================================

class ReportData(BaseModel):
    short_summary: str = Field(
        description=(
            "A concise 2-3 sentence summary of the findings."
        )
    )

    markdown_report: str = Field(
        description="The complete research report in Markdown."
    )

    follow_up_questions: list[str] = Field(
        description=(
            "Important questions or topics that could "
            "be researched further."
        )
    )


WRITER_INSTRUCTIONS = """
You are a senior research analyst.

You will receive:

1. The original research question.
2. Multiple web search summaries.

Use the research to produce a comprehensive,
well-structured report.

The report should:

- Be fact-based.
- Synthesize information from the provided research.
- Clearly organize the important findings.
- Use Markdown formatting.
- Include useful comparisons where appropriate.
- Avoid inventing information.

Produce:

1. A short 2-3 sentence summary.
2. A detailed Markdown report.
3. Follow-up research questions.
"""

writer_agent = Agent(
    name="Writer Agent",
    instructions=WRITER_INSTRUCTIONS,
    model=model,
    output_type=ReportData,
)


# ============================================================
# Agent 4: Email Agent
# ============================================================

@function_tool
def send_email_tool(
    subject: str,
    text_body: str,
    html_body: str,
) -> str:
    """
    Send the final research report by email.

    Args:
        subject: Email subject.
        text_body: Plain-text email body.
        html_body: HTML email body.
    """

    if USE_EMAIL:
        send_email(
            subject,
            text_body,
            html_body,
        )
    else:
        push(
            f"Subject: {subject}\n\n"
            f"{text_body}"
        )

    return "Email sent successfully."


EMAIL_INSTRUCTIONS = """
You are an email assistant.

You are given a detailed research report.

Create a professional email containing the report.

You must:

- Create an appropriate subject.
- Create a plain-text version.
- Create a clean HTML version.
- Use the send_email_tool to send the email.

Do not just describe the email.
Actually call the tool.
"""

email_agent = Agent(
    name="Email Agent",
    instructions=EMAIL_INSTRUCTIONS,
    tools=[send_email_tool],
    model=model,
)


# ============================================================
# Orchestration: Individual Search
# ============================================================

async def search(item: WebSearchItem) -> str:
    """
    Execute one research search using the Search Agent.
    """

    input_message = (
        f"Search term: {item.query}\n"
        f"Reason for searching: {item.reason}"
    )

    result = await Runner.run(
        search_agent,
        input_message,
    )

    return result.final_output


# ============================================================
# Orchestration: Planning + Parallel Search
# ============================================================

async def run_searches(query: str) -> list[str]:
    """
    1. Ask Planner Agent for search queries.
    2. Execute all searches concurrently.
    """

    print("\n[1/3] Planning searches...")

    result = await Runner.run(
        planner_agent,
        f"Query: {query}",
    )

    search_plan = result.final_output
    searches = search_plan.searches

    print(
        f"Planner created {len(searches)} searches."
    )

    # Create search tasks
    tasks = [
        search(item)
        for item in searches
    ]

    # Run searches concurrently
    results = await asyncio.gather(*tasks)

    print("Searches completed.")

    return results


# ============================================================
# Orchestration: Report Generation
# ============================================================

async def write_report(
    query: str,
    search_results: list[str],
) -> ReportData:
    """
    Generate a structured research report.
    """

    print("\n[2/3] Writing report...")

    research = "\n\n--- SEARCH RESULT ---\n\n".join(
        search_results
    )

    input_message = (
        f"Original research question:\n"
        f"{query}\n\n"
        f"Research results:\n"
        f"{research}"
    )

    result = await Runner.run(
        writer_agent,
        input_message,
    )

    print("Report generated.")

    return result.final_output


# ============================================================
# Orchestration: Email
# ============================================================

async def send_report_email(
    report: ReportData,
) -> str:
    """
    Send the generated report using the Email Agent.
    """

    print("\n[3/3] Sending report...")

    result = await Runner.run(
        email_agent,
        report.markdown_report,
    )

    print("Email sent.")

    return result.final_output


# ============================================================
# Main Deep Research Workflow
# ============================================================

async def main():

    query = "Most popular AI Agent frameworks in 2026"

    print("=" * 60)
    print("DEEP RESEARCH AGENT")
    print("=" * 60)

    with trace("Deep Research Workflow"):

        # Step 1:
        # Planner → Search Agents
        search_results = await run_searches(query)

        # Step 2:
        # Search Results → Writer
        report = await write_report(
            query,
            search_results,
        )

        # Step 3:
        # Report → Email Agent
        await send_report_email(report)

    print("\nResearch completed successfully.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())