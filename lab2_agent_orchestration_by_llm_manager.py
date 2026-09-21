import os
import smtplib
import asyncio

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


load_dotenv()


# ============================================================
# Environment Variables
# ============================================================

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")

RECIPIENTS = [
    email.strip()
    for email in os.getenv("RECIPIENTS", "").split(",")
    if email.strip()
]


# ============================================================
# Groq OpenAI-Compatible Client
# ============================================================

groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# Groq Model
# ============================================================

model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=groq_client,
)


# ============================================================
# Common Sales Agent Instructions
# ============================================================

intro = """
You are a sales agent working for ComplAI,
a company that provides a SaaS tool for ensuring
SOC2 compliance and preparing for audits, powered by AI.

You write cold sales emails.

Sender information:

Name: Mazhar Ahmad
Role: Senior Sales Executive
Company: ComplAI
Phone: +91 XXXXX XXXXX
Website: https://complai.com

Never use placeholders such as:
[Your Name]
[Phone]
[Website]
[Your Website]
[your name]
[your phone]
[your website]  
Hi [Recipient’s Name], 

Just say "Hi" or "Hello".

Always use the actual sender information provided above.

The email should sound natural and human.
Do not mention that you are an AI.
Do not invent sender information.
"""


# ============================================================
# Three Different Sales Agents
# ============================================================

instructions1 = intro + """

Your email style is professional, serious,
with gravitas and credibility.

Focus on:
- Trust
- Business value
- SOC2 compliance
- Audit preparation
- Clear and credible communication

Avoid excessive marketing language.
"""


instructions2 = intro + """

Your email style is witty, engaging, and humorous.

Use light humor where appropriate while remaining
professional and credible.

The humor should support the sales message,
not distract from it.

Avoid sounding like a comedy script.
"""


instructions3 = intro + """

Your email style is concise, direct, and to the point,
in the style of a busy senior executive.

Keep the message short.

Focus on:
- The prospect's problem
- The value of ComplAI
- A clear reason to respond

Avoid unnecessary explanations and filler.
"""


sales_agent1 = Agent(
    name="Professional Sales Agent",
    instructions=instructions1,
    model=model,
)


sales_agent2 = Agent(
    name="Humorous Sales Agent",
    instructions=instructions2,
    model=model,
)


sales_agent3 = Agent(
    name="Executive Sales Agent",
    instructions=instructions3,
    model=model,
)


# ============================================================
# Email Sending Function
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
        subject: The subject of the email.
        text_body: The plain-text email body.
        html_body: The HTML email body.
    """

    send_email(
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )

    return "Email sent successfully."


# ============================================================
# Convert Sales Agents into Tools
# ============================================================

description = """
Use this tool to write a cold sales email.

In the input, simply instruct the agent to write
a sales email.

The agent already knows:
- ComplAI's business
- The sender information
- Its assigned writing style
"""


tool1 = sales_agent1.as_tool(
    tool_name="sales_email_writer_1",
    tool_description=description,
)


tool2 = sales_agent2.as_tool(
    tool_name="sales_email_writer_2",
    tool_description=description,
)


tool3 = sales_agent3.as_tool(
    tool_name="sales_email_writer_3",
    tool_description=description,
)


# ============================================================
# Sales Manager
# ============================================================

manager_instructions = """
You are a Sales Manager at ComplAI.

Your job is to create and send the most effective
cold sales email.

Follow this workflow exactly.

1. GENERATE DRAFTS

Use ALL THREE sales email writer tools:

- sales_email_writer_1
- sales_email_writer_2
- sales_email_writer_3

Generate one draft from each agent.

Do not proceed until all three drafts are available.

2. EVALUATE

Review all three drafts.

Compare them based on:

- Clarity
- Professionalism
- Persuasiveness
- Relevance to a potential ComplAI customer
- Natural human tone
- Conciseness
- Strength of the call to action

3. SELECT

Select exactly ONE draft.

Do not combine multiple drafts unless necessary.

4. SEND

Use send_email_tool to send ONLY the selected draft.

Send exactly ONE email.

Never send all three drafts.

The final email must contain:

- Subject
- Plain-text body
- HTML body
"""


tools = [
    tool1,
    tool2,
    tool3,
    send_email_tool,
]


sales_manager = Agent(
    name="Sales Manager",
    instructions=manager_instructions,
    tools=tools,
    model=model,
)


# ============================================================
# Task
# ============================================================

task = """
Create a cold sales email for ComplAI.

Generate three different versions using all three
sales email writer tools.

Review the three drafts.

Select the single strongest draft.

Then use send_email_tool to send only that selected
email to the configured sales recipients.
"""


# ============================================================
# Run
# ============================================================

async def main():

    with trace("Sales Manager"):

        result = await Runner.run(
            sales_manager,
            task,
        )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())