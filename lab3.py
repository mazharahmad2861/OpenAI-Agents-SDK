import os
import asyncio
import smtplib

from email.message import EmailMessage
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    trace,
    output_guardrail,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    ModelSettings,
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
Role:  Sales Executive
Company: ComplAI
Phone: +91 XXXXX XXXXX
Website: https://complai.com

Never use placeholders such as:
[Your Name], [Phone], [Website], [your name], etc. 

Never use placeholders in salutation or an email greeting: 
Dear [Name], or Hello [Name], or Hi [Recipient's Name] ; just say hi or hello or greetings, etc.

Always use the actual sender information above.

Never mention that you are an AI.
Do not invent sender information.

Return only the email.
"""


# ============================================================
# Agent Instructions
# ============================================================

professional_instructions = """
Write in a professional, serious style.

Focus on trust, business value, SOC2 compliance,
and audit preparation.

Avoid excessive marketing language.
"""

humorous_instructions = """
Write in a witty and engaging style.

Use light humor while remaining professional
and credible.

Do not let the humor distract from the sales message.
"""

executive_instructions = """
Write in a concise and direct executive style.

Focus on the prospect's problem, the value of ComplAI,
and a clear call to action.

Avoid filler and unnecessary explanations.
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
# Convert Agents → Tools
# ============================================================

writer_description = """
Write one cold sales email for ComplAI.

The email must contain:
- A natural greeting
- A concise sales message
- A clear call to action
- The actual sender information

Return only the email.
"""

professional_tool = professional_agent.as_tool(
    tool_name="professional_sales_writer",
    tool_description=writer_description,
)

humorous_tool = humorous_agent.as_tool(
    tool_name="humorous_sales_writer",
    tool_description=writer_description,
)

executive_tool = executive_agent.as_tool(
    tool_name="executive_sales_writer",
    tool_description=writer_description,
)


# ============================================================
# Output Guardrail
# ============================================================

@output_guardrail
async def email_guardrail(ctx, agent, message):

    email = str(message)

    forbidden_placeholders = [
        "[Your Name]",
        "[Phone]",
        "[Website]",
        "[your name]",
        "[phone]",
        "[website]",
        "[Name]",
        "[First Name]",
        "[Company]",
    ]

    contains_placeholder = any(
        placeholder in email
        for placeholder in forbidden_placeholders
    )

    required_sender_info = [
        "Mazhar Ahmad",
        "Sales Executive",
        "ComplAI",
        "+91 XXXXX XXXXX",
        "https://complai.com",
    ]

    missing_sender_info = [
        item
        for item in required_sender_info
        if item not in email
    ]

    is_problem = (
        not email.strip()
        or contains_placeholder
        or bool(missing_sender_info)
    )

    return GuardrailFunctionOutput(
        output_info={
            "contains_placeholders": contains_placeholder,
            "missing_sender_info": missing_sender_info,
        },
        tripwire_triggered=is_problem,
    )


# ============================================================
# Sales Manager
# ============================================================

manager_instructions = """
You are the Sales Manager at ComplAI.

Create three different sales email drafts.

You MUST use all three writer tools:

1. professional_sales_writer
2. humorous_sales_writer
3. executive_sales_writer

After receiving the three drafts:

- Compare them.
- Select the strongest one.
- Prefer clarity, professionalism, relevance,
  natural tone, conciseness, and a strong CTA.

Return ONLY the selected email.

Do not explain your selection.
Do not send an email.
"""


sales_manager = Agent(
    name="Sales Manager",
    instructions=manager_instructions,
    tools=[
        professional_tool,
        humorous_tool,
        executive_tool,
    ],
    model=model,

    # Important for Groq gpt-oss-20b
    model_settings=ModelSettings(
        parallel_tool_calls=False,
    ),

    output_guardrails=[
        email_guardrail,
    ],
)


# ============================================================
# SMTP Email Sender
# ============================================================

def send_email(
    subject: str,
    text_body: str,
    html_body: str,
):

    if not EMAIL_ADDRESS:
        raise ValueError("EMAIL_ADDRESS is missing.")

    if not EMAIL_APP_PASSWORD:
        raise ValueError("EMAIL_APP_PASSWORD is missing.")

    if not EMAIL_SMTP_SERVER:
        raise ValueError("EMAIL_SMTP_SERVER is missing.")

    if not RECIPIENTS:
        raise ValueError("RECIPIENTS is empty.")

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
        timeout=30,
    ) as server:

        server.starttls()

        server.login(
            EMAIL_ADDRESS,
            EMAIL_APP_PASSWORD,
        )

        server.send_message(msg)


# ============================================================
# Task
# ============================================================

task = """
Create a cold sales email for ComplAI.

Generate:

1. One professional version
2. One humorous version
3. One executive version

Compare the three drafts and return ONLY
the strongest final email.
"""


# ============================================================
# Main
# ============================================================

async def main():

    with trace("ComplAI Sales Automation"):

        try:

            result = await Runner.run(
                sales_manager,
                task,
            )

            email = result.final_output

            print("\n--- APPROVED EMAIL ---\n")
            print(email)

            # ==================================================
            # SEND EMAIL ONLY AFTER GUARDRAIL PASSES
            # ==================================================

            send_email(
                subject="Simplifying SOC2 Compliance with ComplAI",
                text_body=email,
                html_body=f"""
<html>
<body>
<p>{email.replace(chr(10), "<br>")}</p>
</body>
</html>
""",
            )

            print("\n--- EMAIL SENT SUCCESSFULLY ---")
            print(f"Recipients: {', '.join(RECIPIENTS)}")

        except OutputGuardrailTripwireTriggered:

            print("\n--- EMAIL BLOCKED ---")
            print("The generated email failed the guardrail.")

        except Exception as e:

            print("\n--- ERROR ---")
            print(f"{type(e).__name__}: {e}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())