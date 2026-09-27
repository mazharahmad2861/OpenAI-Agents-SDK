import os
import asyncio
import smtplib
from email.message import EmailMessage
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import TavilyClient

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    ModelSettings,
    function_tool,
    trace,
    set_tracing_disabled,
)
from pydantic import BaseModel, Field


# ============================================================
# Configuration
# ============================================================

load_dotenv(override=True)

GROQ_MODEL = "openai/gpt-oss-20b"
HOW_MANY_SEARCHES = 5
USE_EMAIL = False  # Change to True after configuring email values


# ============================================================
# Clients
# ============================================================

groq_client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

tavily_client = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)

model = OpenAIChatCompletionsModel(
    model=GROQ_MODEL,
    openai_client=groq_client,
)


# ============================================================
# Email function
# ============================================================

def send_email(
    subject: str,
    text_body: str,
    html_body: str,
) -> None:
    msg = EmailMessage()
    msg["From"] = os.environ["EMAIL_ADDRESS"]
    msg["To"] = ", ".join(
        email.strip()
        for email in os.environ["RECIPIENTS"].split(",")
        if email.strip()
    )
    msg["Subject"] = subject

    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(
        os.environ["EMAIL_SMTP_SERVER"],
        587,
        timeout=30,
    ) as server:
        server.starttls()
        server.login(
            os.environ["EMAIL_ADDRESS"],
            os.environ["EMAIL_APP_PASSWORD"],
        )
        server.send_message(msg)


@function_tool
def send_email_tool(
    subject: str,
    text_body: str,
    html_body: str,
) -> str:
    """Send the final research report by email."""

    if not USE_EMAIL:
        return "Email is disabled. Report was generated but not sent."

    send_email(subject, text_body, html_body)
    return "Email sent successfully."


# ============================================================
# Structured output models
# ============================================================

class WebSearchItem(BaseModel):
    reason: str = Field(
        description="Why this search is useful for answering the query."
    )
    query: str = Field(
        description="The exact web search query to run."
    )


class WebSearchPlan(BaseModel):
    searches: List[WebSearchItem] = Field(
        description="A list of useful web searches."
    )


class ReportData(BaseModel):
    short_summary: str = Field(
        description="A concise 2-3 sentence summary."
    )
    markdown_report: str = Field(
        description="The complete research report in Markdown."
    )
    follow_up_questions: List[str] = Field(
        description="Useful questions for further research."
    )


# ============================================================
# Tavily search tool
# ============================================================

@function_tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily.

    Args:
        query: The search query.
    """
    response = tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer=False,
    )

    results = response.get("results", [])

    if not results:
        return "No results found."

    formatted_results = []

    for index, result in enumerate(results, start=1):
        formatted_results.append(
            f"Result {index}\n"
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('content', '')}"
        )

    return "\n\n".join(formatted_results)


# ============================================================
# Agents
# ============================================================

planner_agent = Agent(
    name="Planner Agent",
    instructions=f"""
You are a research planning agent.

Given a user query, create exactly {HOW_MANY_SEARCHES} useful and
different web searches. Cover different aspects of the query.
Do not answer the query. Only produce the structured search plan.
""",
    model=model,
    output_type=WebSearchPlan,
)

search_agent = Agent(
    name="Search Agent",
    instructions="""
You are a web research assistant.

Use the web_search tool to investigate the provided search term.
Summarize the most relevant findings in fewer than 300 words.
Include source URLs from the search results.
Do not invent facts or sources.
""",
    model=model,
    model_settings=ModelSettings(tool_choice="required"),
    tools=[web_search],
)

writer_agent = Agent(
    name="Writer Agent",
    instructions="""
You are a senior research writer.

Use the original query and the collected research to write a
clear, accurate, well-structured Markdown report.

Requirements:
- Include a short executive summary.
- Use headings and bullet points where useful.
- Distinguish facts from uncertainty.
- Do not invent information.
- Include source URLs supplied in the research.
- Add useful follow-up research questions.
""",
    model=model,
    output_type=ReportData,
)

email_agent = Agent(
    name="Email Agent",
    instructions="""
You are an email formatting assistant.

You will receive a Markdown research report.
Create a professional email subject, plain-text body, and HTML body.
Then use the send_email_tool exactly once.

If email sending is disabled, report that clearly.
""",
    model=model,
    tools=[send_email_tool],
)


# ============================================================
# Workflow functions
# ============================================================

async def create_search_plan(query: str) -> WebSearchPlan:
    print("Planning searches...")
    result = await Runner.run(
        planner_agent,
        f"Research query: {query}",
    )
    return result.final_output


async def run_one_search(item: WebSearchItem) -> str:
    prompt = (
        f"Search term: {item.query}\n"
        f"Reason: {item.reason}"
    )

    result = await Runner.run(search_agent, prompt)
    return result.final_output


async def run_searches(query: str) -> List[str]:
    plan = await create_search_plan(query)

    print(f"Running {len(plan.searches)} searches...")
    results = await asyncio.gather(
        *(run_one_search(item) for item in plan.searches)
    )

    print("Searches completed.")
    return results


async def write_report(
    query: str,
    search_results: List[str],
) -> ReportData:
    print("Writing report...")

    research_text = "\n\n---\n\n".join(search_results)

    result = await Runner.run(
        writer_agent,
        (
            f"Original query:\n{query}\n\n"
            f"Research results:\n{research_text}"
        ),
    )

    print("Report completed.")
    return result.final_output


async def send_report_email(report: ReportData) -> str:
    print("Preparing email...")

    result = await Runner.run(
        email_agent,
        report.markdown_report,
    )

    print("Email step completed.")
    return result.final_output


# ============================================================
# Main
# ============================================================

async def main() -> None:
    query = "Most popular AI agent frameworks in 2026"

    with trace("Tavily Deep Research"):
        print("Starting research...\n")

        search_results = await run_searches(query)
        report = await write_report(query, search_results)

        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70)
        print(report.markdown_report)

        print("\nFOLLOW-UP QUESTIONS")
        for question in report.follow_up_questions:
            print(f"- {question}")

        if USE_EMAIL:
            await send_report_email(report)


if __name__ == "__main__":
    set_tracing_disabled(True)
    asyncio.run(main())
