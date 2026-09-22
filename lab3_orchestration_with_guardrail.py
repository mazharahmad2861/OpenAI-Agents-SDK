import os
import asyncio
import smtplib

from email.message import EmailMessage
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agents import (
    Agent,
    Runner,
    function_tool,
    trace,
    output_guardrail,
    GuardrailFunctionOutput,
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
# Structured Output
# ============================================================

class EmailReview(BaseModel):

    is_professional: bool = Field(
        description="Whether the email is professional and appropriate."
    )

    number_of_sentences: int = Field(
        description=(
            "Number of sentences in the email body, "
            "excluding the greeting and signature."
        )
    )

    contains_placeholders: bool = Field(
        description=(
            "Whether the email contains placeholders such as "
            "[Name], [Phone], [Website], etc."
        )
    )


# ============================================================
# Email Checker
# ============================================================

checker = Agent(
    name="Email Checker",
    instructions="""
You review cold sales emails.

Check whether the email:

1. Is professional and appropriate.
2. Contains placeholders such as:
   [Name], [First Name], [Company], [Phone], [Website].
3. Count the number of sentences in the email body,
   excluding the greeting and signature.

Return the result using the EmailReview structured output.
""",
    model=model,
    output_type=EmailReview,
)


# ============================================================
# Output Guardrail
# ============================================================

@output_guardrail
async def email_guardrail(ctx, agent, message):

    result = await Runner.run(
        checker,
        message,
        context=ctx.context,
    )

    review = result.final_output

    is_problem = (
        review.contains_placeholders
        or not review.is_professional
    )

    return GuardrailFunctionOutput(
        output_info={
            "review": review,
        },
        tripwire_triggered=is_problem,
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
# Email Sending Tool
# ============================================================

@function_tool
def send_email_tool(
    subject: str,
    text_body: str,
    html_body: str,
) -> str:
    """
    Send the selected sales email to all configured recipients.
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

writer_description = """
Use this tool to write a cold sales email for ComplAI.

Simply instruct the agent to write a sales email.
The agent already knows the company information,
sender information, and its assigned writing style.
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
# Sales Manager
# ============================================================

manager_instructions = """
You are a Sales Manager at ComplAI.

Your goal is to find the single best cold sales email.

Follow these steps:

1. Generate Drafts

Use ALL THREE sales writer tools:

- professional_sales_writer
- humorous_sales_writer
- executive_sales_writer

Generate one draft from each agent.

2. Evaluate and Select

Review all three drafts.

Select exactly ONE draft based on:

- Clarity
- Professionalism
- Persuasiveness
- Relevance
- Natural tone
- Conciseness
- Strength of call to action

3. Return ONLY the selected email.

Do not send the email yourself.
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
    output_guardrails=[
        email_guardrail,
    ],
)


# ============================================================
# Task
# ============================================================

task = """
Create a cold sales email for ComplAI.

Generate three different drafts using:

1. Professional Sales Writer
2. Humorous Sales Writer
3. Executive Sales Writer

Review the drafts and select the single strongest email.

Return only the selected email.
"""


# ============================================================
# Run
# ============================================================

async def main():

    with trace("Sales Manager - Tools + Structured Output + Guardrail"):

        try:

            result = await Runner.run(
                sales_manager,
                task,
            )

            email = result.final_output

            print("\n--- APPROVED EMAIL ---\n")
            print(email)

        except Exception as e:

            print("\n--- EMAIL BLOCKED ---\n")
            print(f"Reason: {e}")


if __name__ == "__main__":
    asyncio.run(main())