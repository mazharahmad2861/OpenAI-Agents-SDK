import os
import asyncio
import smtplib

from email.message import EmailMessage
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    function_tool,
    trace,
    OpenAIChatCompletionsModel,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()



EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")

RECIPIENTS = [
    email.strip()
    for email in os.getenv("RECIPIENTS", "").split(",")
    if email.strip()
]


# ============================================================
# Groq Model
# ============================================================

groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=groq_client,
)


# ============================================================
# Common Instructions
# ============================================================

intro = """
You are a sales agent working for ComplAI,
a company that provides a SaaS tool for ensuring SOC2
compliance and preparing for audits, powered by AI.

You write cold sales emails.

Sender information:
Name: Mazhar Ahmad
Role: Senior Sales Executive
Company: ComplAI
Phone: +91 XXXXX XXXXX
Website: https://complai.com

Never use placeholders such as:
[Your Name], [Phone], [Website], [your name], etc.

Always use the actual sender information provided above.

Never mention that you are an AI.
Do not invent sender information.
"""


# ============================================================
# Agent-Specific Instructions
# ============================================================

professional_instructions = """
Your email style is professional, serious,
with gravitas and credibility.

Focus on trust, business value, SOC2 compliance,
and audit preparation.

Avoid excessive marketing language.
"""


humorous_instructions = """
Your email style is witty, engaging, and humorous.

Use light humor where appropriate while remaining
professional and credible.

Do not let the humor distract from the sales message.
"""


executive_instructions = """
Your email style is concise, direct, and to the point,
in the style of a busy senior executive.

Focus on the prospect's problem, the value of ComplAI,
and a clear call to action.

Avoid unnecessary explanations and filler.
"""


# ============================================================
# Sales Writer Agents
# ============================================================

professional_agent = Agent(
    name="Professional Sales Agent",
    instructions=intro + professional_instructions,
    model=model,
)


humorous_agent = Agent(
    name="Humorous Sales Agent",
    instructions=intro + humorous_instructions,
    model=model,
)


executive_agent = Agent(
    name="Executive Sales Agent",
    instructions=intro + executive_instructions,
    model=model,
)


# ============================================================
# Email Function
# ============================================================

def send_email(
    subject: str,
    text_body: str,
    html_body: str,
):
    msg = EmailMessage()

    msg["From"] = EMAIL_ADDRESS
    msg["To"] = ", ".join(RECIPIENTS)
    msg["Subject"] = subject

    msg.set_content(text_body)
    msg.add_alternative(
        html_body,
        subtype="html",
    )

    with smtplib.SMTP(
        EMAIL_SMTP_SERVER,
        587,
    ) as server:
        server.starttls()

        server.login(
            EMAIL_ADDRESS,
            EMAIL_APP_PASSWORD,
        )

        server.send_message(msg)


# ============================================================
# Email Tool
# ============================================================

@function_tool
def send_email_tool(
    subject: str,
    text_body: str,
    html_body: str,
) -> str:
    """
    Send the selected sales email to all configured recipients.

    Args:
        subject: Email subject.
        text_body: Plain-text email body.
        html_body: HTML email body.
    """

    send_email(
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )

    return "Email sent successfully."


# ============================================================
# Convert Writer Agents into Tools
# ============================================================

description = """
Use this tool to write a cold sales email for ComplAI.

Simply instruct the agent to write a sales email.
The agent already knows the company information,
sender information, and its assigned writing style.
"""


professional_tool = professional_agent.as_tool(
    tool_name="professional_sales_writer",
    tool_description=description,
)


humorous_tool = humorous_agent.as_tool(
    tool_name="humorous_sales_writer",
    tool_description=description,
)


executive_tool = executive_agent.as_tool(
    tool_name="executive_sales_writer",
    tool_description=description,
)


# ============================================================
# Sales Sender
# ============================================================

sales_sender = Agent(
    name="Sales Sender",
    instructions="""
You are the Sales Sender at ComplAI.

You will receive three cold sales email drafts
from the Sales Manager.

Review all three drafts and select exactly ONE.

Evaluate them based on:

- Clarity
- Professionalism
- Persuasiveness
- Relevance
- Natural tone
- Conciseness
- Call to action

After selecting the best draft, use send_email_tool
to send ONLY that email.

Send exactly ONE email.

Do not send the other drafts.
""",
    tools=[send_email_tool],
    model=model,
)


# ============================================================
# Sales Manager
# ============================================================

manager_instructions = """
You are a Sales Manager at ComplAI.

Your job is to coordinate the sales email creation process.

Follow these steps exactly:

1. Generate Drafts

Use ALL THREE sales writer tools:

- professional_sales_writer
- humorous_sales_writer
- executive_sales_writer

Generate one email from each agent.

Do not proceed until all three drafts are ready.

2. Handoff

Once all three drafts are ready,
handoff to the Sales Sender.

The Sales Sender will review the drafts,
select the best one, and send it.

Do not send an email yourself.
"""


sales_manager = Agent(
    name="Sales Manager",
    instructions=manager_instructions,
    tools=[
        professional_tool,
        humorous_tool,
        executive_tool,
    ],
    handoffs=[
        sales_sender,
    ],
    model=model,
)


# ============================================================
# Task
# ============================================================

task = """
Create a cold sales email for ComplAI.

Generate three different versions:

1. Professional
2. Humorous
3. Executive

Use all three sales writer tools.

Once all three drafts are ready,
handoff to the Sales Sender.

The Sales Sender will review the drafts,
select one, and send only that email.
"""


# ============================================================
# Run
# ============================================================

async def main():

    with trace("Sales Manager - Handoff Workflow"):

        result = await Runner.run(
            sales_manager,
            task,
        )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())