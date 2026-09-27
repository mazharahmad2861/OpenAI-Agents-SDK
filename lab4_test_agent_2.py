import os
import asyncio
import smtplib
import json
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
# CONFIGURATION
# ============================================================

load_dotenv(override=True)

GROQ_MODEL = "openai/gpt-oss-20b"

HOW_MANY_SEARCHES = 5

USE_EMAIL = False


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]


# ============================================================
# CLIENTS
# ============================================================

groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY,
)


# ============================================================
# MODEL
# ============================================================

model = OpenAIChatCompletionsModel(
    model=GROQ_MODEL,
    openai_client=groq_client,
)


# ============================================================
# EMAIL
# ============================================================

def send_email(
    subject: str,
    text_body: str,
    html_body: str,
) -> None:

    msg = EmailMessage()

    msg["From"] = os.environ["EMAIL_ADDRESS"]

    recipients = [
        email.strip()
        for email in os.environ["RECIPIENTS"].split(",")
        if email.strip()
    ]

    msg["To"] = ", ".join(recipients)

    msg["Subject"] = subject

    msg.set_content(text_body)

    msg.add_alternative(
        html_body,
        subtype="html",
    )

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
    """
    Send the final research report by email.
    """

    if not USE_EMAIL:
        return (
            "Email is disabled. "
            "The report was generated but not sent."
        )

    send_email(
        subject,
        text_body,
        html_body,
    )

    return "Email sent successfully."


# ============================================================
# PYDANTIC MODELS
# ============================================================

class WebSearchItem(BaseModel):

    reason: str = Field(
        description=(
            "Why this search is useful for "
            "answering the research query."
        )
    )

    query: str = Field(
        description=(
            "The exact web search query "
            "to execute."
        )
    )


class WebSearchPlan(BaseModel):

    searches: List[WebSearchItem] = Field(
        description=(
            "List of web searches required "
            "for the research."
        )
    )


class ReportData(BaseModel):

    short_summary: str = Field(
        description=(
            "A concise 2-3 sentence summary "
            "of the research."
        )
    )

    markdown_report: str = Field(
        description=(
            "The complete research report "
            "written in Markdown."
        )
    )

    follow_up_questions: List[str] = Field(
        description=(
            "Useful questions for further research."
        )
    )


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_output(
    output: str,
) -> dict:

    if not isinstance(output, str):

        raise TypeError(
            "Expected model output to be a string, "
            f"got {type(output).__name__}"
        )

    cleaned = output.strip()

    # --------------------------------------------------------
    # Remove ```json
    # --------------------------------------------------------

    if cleaned.startswith("```json"):

        cleaned = cleaned[
            len("```json"):
        ].strip()

    # --------------------------------------------------------
    # Remove ```
    # --------------------------------------------------------

    elif cleaned.startswith("```"):

        cleaned = cleaned[
            len("```"):
        ].strip()

    # --------------------------------------------------------
    # Remove closing ```
    # --------------------------------------------------------

    if cleaned.endswith("```"):

        cleaned = cleaned[
            :-3
        ].strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        return json.loads(cleaned)

    except json.JSONDecodeError as error:

        print("\n" + "=" * 70)
        print("INVALID JSON FROM MODEL")
        print("=" * 70)

        print(cleaned)

        print("=" * 70)

        raise ValueError(
            f"Invalid JSON returned by model: {error}"
        ) from error


# ============================================================
# TAVILY WEB SEARCH TOOL
# ============================================================

@function_tool
def web_search(
    query: str,
) -> str:
    """
    Search the internet using Tavily.

    IMPORTANT:
    This tool accepts exactly one argument:

    query: string
    """

    print(
        f"\n[TAVILY] Searching: {query}"
    )

    response = tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer=False,
    )

    results = response.get(
        "results",
        [],
    )

    if not results:

        return (
            "No search results were found "
            f"for query: {query}"
        )

    formatted_results = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        title = result.get(
            "title",
            "",
        )

        url = result.get(
            "url",
            "",
        )

        content = result.get(
            "content",
            "",
        )

        formatted_results.append(
            f"""
Result {index}

Title:
{title}

URL:
{url}

Content:
{content}
"""
        )

    return "\n".join(
        formatted_results
    )


# ============================================================
# PLANNER AGENT
# ============================================================

planner_agent = Agent(

    name="Planner Agent",

    instructions=f"""
You are a research planning agent.

Your job is to create a research plan
for the user's research question.

Create EXACTLY {HOW_MANY_SEARCHES}
different web search queries.

Each query should investigate a
different aspect of the topic.

Return ONLY valid JSON.

The JSON MUST have this structure:

{{
    "searches": [
        {{
            "reason": "why this search is useful",
            "query": "exact search query"
        }}
    ]
}}

Rules:

1. Create exactly {HOW_MANY_SEARCHES} searches.
2. Every search must have "reason".
3. Every search must have "query".
4. "query" must be a string.
5. Do not use Markdown.
6. Do not use ```json.
7. Do not add explanations outside JSON.
""",

    model=model,
)


# ============================================================
# SEARCH AGENT
# ============================================================

search_agent = Agent(

    name="Search Agent",

    instructions="""
You are a web research agent.

You have access to ONE tool:

web_search

The tool accepts EXACTLY ONE parameter:

query: string

The tool call MUST look like:

{
    "query": "your search query"
}

IMPORTANT:

NEVER send:

{
    "cursor": 1,
    "id": 1
}

NEVER use:

- cursor
- id
- page
- offset
- limit
- results
- search_id
- any other parameter

ONLY use:

query

Example:

web_search(
    query="AI agent frameworks 2026"
)

Your workflow:

1. Read the research task.
2. Determine the exact search query.
3. Call web_search.
4. Analyze the returned search results.
5. Extract relevant information.
6. Include source URLs.
7. Do not invent facts.
8. Do not invent URLs.

You MUST call web_search before
giving your final answer.
""",

    model=model,

    model_settings=ModelSettings(
        tool_choice="required",
    ),

    tools=[
        web_search,
    ],
)


# ============================================================
# WRITER AGENT
# ============================================================

writer_agent = Agent(

    name="Writer Agent",

    instructions="""
You are a senior research writer.

You will receive:

1. The original research question.
2. Results produced by multiple
   Search Agent executions.

Use ONLY the supplied research.

Create a detailed research report.

Return ONLY valid JSON.

Required structure:

{
    "short_summary": "2-3 sentence summary",

    "markdown_report": "complete Markdown report",

    "follow_up_questions": [
        "question 1",
        "question 2",
        "question 3"
    ]
}

Rules:

1. Do not use ```json.
2. Do not add text outside JSON.
3. Do not invent facts.
4. Do not invent sources.
5. Include useful source URLs.
6. Clearly distinguish factual findings
   from opinions or claims.
7. The markdown_report should be
   complete and readable.
""",

    model=model,
)


# ============================================================
# EMAIL AGENT
# ============================================================

email_agent = Agent(

    name="Email Agent",

    instructions="""
You are an email formatting agent.

You will receive a research report.

Your job is to:

1. Create a professional email subject.
2. Create a plain-text email body.
3. Create an HTML email body.
4. Call send_email_tool exactly once.

If email sending is disabled,
clearly report that the email was
not sent.

Do not modify the research facts.
""",

    model=model,

    tools=[
        send_email_tool,
    ],
)


# ============================================================
# CREATE SEARCH PLAN
# ============================================================

async def create_search_plan(
    query: str,
) -> WebSearchPlan:

    print(
        "Planning searches..."
    )

    result = await Runner.run(

        planner_agent,

        f"""
Research query:

{query}
""",
    )

    print(
        "\nPlanner output:"
    )

    print(
        result.final_output
    )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    data = parse_json_output(
        result.final_output
    )

    # --------------------------------------------------------
    # Validate with Pydantic
    # --------------------------------------------------------

    plan = WebSearchPlan.model_validate(
        data
    )

    # --------------------------------------------------------
    # Check search count
    # --------------------------------------------------------

    if len(plan.searches) != HOW_MANY_SEARCHES:

        raise ValueError(
            f"Planner returned "
            f"{len(plan.searches)} searches. "
            f"Expected {HOW_MANY_SEARCHES}."
        )

    return plan


# ============================================================
# RUN ONE SEARCH AGENT
# ============================================================

async def run_one_search(
    item: WebSearchItem,
) -> str:

    print(
        f"\nSearching: {item.query}"
    )

    prompt = f"""
You must research the following topic.

SEARCH QUERY:
{item.query}

REASON:
{item.reason}

IMPORTANT:

Call the web_search tool.

The tool has exactly one argument:

query

Use this exact query:

{item.query}

The tool call must contain ONLY:

{{
    "query": "{item.query}"
}}

Do NOT use cursor.
Do NOT use id.
Do NOT use page.
Do NOT use offset.
Do NOT use any other arguments.

After receiving the search results,
summarize the relevant findings
and include the source URLs.
"""

    try:

        result = await Runner.run(
            search_agent,
            prompt,
        )

        return result.final_output

    except Exception as error:

        print(
            "\nSEARCH AGENT ERROR"
        )

        print(
            f"Query: {item.query}"
        )

        print(
            f"Error: {error}"
        )

        raise


# ============================================================
# RUN ALL SEARCHES
# ============================================================

async def run_searches(
    query: str,
) -> List[str]:

    plan = await create_search_plan(
        query
    )

    print(
        f"\nRunning {len(plan.searches)} searches..."
    )

    tasks = [
        run_one_search(item)
        for item in plan.searches
    ]

    results = await asyncio.gather(
        *tasks
    )

    print(
        "\nAll searches completed."
    )

    return results


# ============================================================
# WRITE REPORT
# ============================================================

async def write_report(
    query: str,
    search_results: List[str],
) -> ReportData:

    print(
        "\nWriting report..."
    )

    research_text = "\n\n".join(

        f"""
==================================================
RESEARCH RESULT {index}
==================================================

{result}
"""
        for index, result
        in enumerate(
            search_results,
            start=1,
        )
    )

    prompt = f"""
ORIGINAL RESEARCH QUERY:

{query}


RESEARCH RESULTS:

{research_text}


Using the research results above,
write the final research report.

Return only the required JSON.
"""

    result = await Runner.run(
        writer_agent,
        prompt,
    )

    print(
        "\nWriter output:"
    )

    print(
        result.final_output
    )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    data = parse_json_output(
        result.final_output
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    report = ReportData.model_validate(
        data
    )

    print(
        "\nReport completed."
    )

    return report


# ============================================================
# SEND REPORT EMAIL
# ============================================================

async def send_report_email(
    report: ReportData,
) -> str:

    print(
        "\nPreparing email..."
    )

    result = await Runner.run(

        email_agent,

        report.markdown_report,
    )

    print(
        "\nEmail step completed."
    )

    return result.final_output


# ============================================================
# MAIN WORKFLOW
# ============================================================

async def main() -> None:

    query = (
        "Most popular AI agent frameworks in 2026"
    )

    with trace(
        "Tavily Multi-Agent Deep Research"
    ):

        print(
            "Starting research...\n"
        )

        # ====================================================
        # 1. PLANNER
        # ====================================================

        search_results = await run_searches(
            query
        )

        # ====================================================
        # 2. WRITER
        # ====================================================

        report = await write_report(
            query,
            search_results,
        )

        # ====================================================
        # 3. FINAL REPORT
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL REPORT"
        )

        print(
            "=" * 70
        )

        print(
            report.markdown_report
        )

        # ====================================================
        # 4. FOLLOW-UP QUESTIONS
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FOLLOW-UP QUESTIONS"
        )

        print(
            "=" * 70
        )

        for question in report.follow_up_questions:

            print(
                f"- {question}"
            )

        # ====================================================
        # 5. EMAIL
        # ====================================================

        if USE_EMAIL:

            email_result = await send_report_email(
                report
            )

            print(
                "\nEmail result:"
            )

            print(
                email_result
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # We are using Groq instead of OpenAI,
    # so disable Agents SDK tracing.

    set_tracing_disabled(True)

    asyncio.run(
        main()
    )

    # use different model, rate limit exceeded, need more tokens, upgrade to dev tier