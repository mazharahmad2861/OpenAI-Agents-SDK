import os
import asyncio
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    ModelSettings,
    OpenAIChatCompletionsModel,
    function_tool,
    trace,
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
# 1. SALES AGENTS
# ============================================================

intro = """
You are a sales agent working for ComplAI,
a company that provides a SaaS tool for ensuring SOC2 compliance
and preparing for audits, powered by AI.
You write cold sales emails.
"""


sales_agent1 = Agent(
    name="Professional Sales Agent",
    instructions=intro + """
Your email style is professional, serious,
with gravitas and credibility.
""",
    model=model,
)


sales_agent2 = Agent(
    name="Humorous Sales Agent",
    instructions=intro + """
Your email style is witty, engaging, and humorous.
""",
    model=model,
)


sales_agent3 = Agent(
    name="Executive Sales Agent",
    instructions=intro + """
Your email style is concise and to the point,
in the style of a busy senior executive.
""",
    model=model,
)


# ============================================================
# 2. SALES PICKER
# ============================================================

sales_picker = Agent(
    name="Sales Picker",
    instructions="""
You pick the best cold sales email from the given options.

Imagine you are a customer and pick the one
you are most likely to respond to.

Do not give an explanation.
Reply with the selected email only.
""",
    model=model,
)


# ============================================================
# 3. EMAIL SENDING
# ============================================================

def send_email(subject, text_body, html_body):

    for recipient in RECIPIENTS:

        msg = EmailMessage()

        msg["From"] = EMAIL_ADDRESS
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.set_content(text_body)
        msg.add_alternative(
            html_body,
            subtype="html",
        )

        with smtplib.SMTP(EMAIL_SMTP_SERVER, 587) as server:
            server.starttls()
            server.login(
                EMAIL_ADDRESS,
                EMAIL_APP_PASSWORD,
            )
            server.send_message(msg)


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
        text_body: The plain-text body of the email.
        html_body: The HTML body of the email.
    """

    send_email(
        subject,
        text_body,
        html_body,
    )

    return "Email sent successfully"


# ============================================================
# 4. SALES SENDER
# ============================================================

sales_sender = Agent(
    name="Sales Sender",
    instructions="""
Use the selected sales email to send the email.

You must use the send_email_tool.
""",
    model=model,
    tools=[send_email_tool],
    model_settings=ModelSettings(
        tool_choice="required"
    ),
)


# ============================================================
# 5. ORCHESTRATION
# ============================================================

async def main():

    message = "Write a cold sales email"

    with trace("Sales Selection Workflow"):

        # Generate 3 emails in parallel
        results = await asyncio.gather(
            Runner.run(sales_agent1, message),
            Runner.run(sales_agent2, message),
            Runner.run(sales_agent3, message),
        )

        # Collect the generated emails
        outputs = [
            result.final_output
            for result in results
        ]

        # Display all generated emails
        print("\n" + "=" * 60)
        print("GENERATED SALES EMAILS")
        print("=" * 60)

        for i, output in enumerate(outputs, 1):
            print(f"\n--- Email {i} ---\n")
            print(output)

        # Prepare emails for Sales Picker
        emails = (
            "Cold sales emails:\n\n"
            + "\n\nEmail:\n\n".join(outputs)
        )

        # Sales Picker selects the best email
        picked = await Runner.run(
            sales_picker,
            emails,
        )

        print("\n" + "=" * 60)
        print("SELECTED EMAIL")
        print("=" * 60)
        print(f"\n{picked.final_output}")

        # Sales Sender sends the selected email
        response = await Runner.run(
            sales_sender,
            picked.final_output,
        )

        print("\n" + "=" * 60)
        print("SENDER RESPONSE")
        print("=" * 60)
        print(f"\n{response.final_output}")


if __name__ == "__main__":
    asyncio.run(main())