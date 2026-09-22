import os
import asyncio
import smtplib
import json

from email.message import EmailMessage

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    function_tool,
    trace,
    output_guardrail,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
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
# Groq Client
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
# STRUCTURED OUTPUT MODELS
# ============================================================

class SalesEmail(BaseModel):

    subject: str = Field(
        description="Email subject line."
    )

    text_body: str = Field(
        description="Plain-text version of the email."
    )

    html_body: str = Field(
        description="HTML version of the email."
    )


class EmailReview(BaseModel):

    is_professional: bool = Field(
        description=(
            "Whether the email is professional, "
            "appropriate and suitable for a cold sales email."
        )
    )

    number_of_sentences: int = Field(
        description=(
            "Number of sentences in the email body, "
            "excluding greeting and signature."
        )
    )

    contains_placeholders: bool = Field(
        description=(
            "Whether the email contains placeholders such as "
            "[Name], [Phone], [Website], etc."
        )
    )


# ============================================================
# COMMON SALES AGENT INSTRUCTIONS
# ============================================================

intro = """
You are a sales agent working for ComplAI,
a company that provides a SaaS tool for SOC2 compliance
and audit preparation, powered by AI.

You write cold sales emails.

Sender information:

Name: Mazhar Ahmad
Role: Senior Sales Executive
Company: ComplAI
Phone: +91 XXXXX XXXXX
Website: https://complai.com

IMPORTANT:

Never use placeholders such as:

[Your Name]
[Phone]
[Website]
[Your Website]
[your name]
[your phone]
[your website]
[Recipient Name]
[First Name]

Always use the actual sender information.

For the recipient greeting use only:

Hi

or

Hello

Never invent a recipient name.

The email must sound natural and human.

Do not mention that you are an AI.

Do not invent sender information.
"""


# ============================================================
# THREE SALES AGENTS
# ============================================================

instructions1 = intro + """

Your style is professional, serious,
credible and business-focused.

Focus on:

- Trust
- Business value
- SOC2 compliance
- Audit preparation
- Credibility

Avoid excessive marketing language.
"""


instructions2 = intro + """

Your style is witty, engaging and lightly humorous.

Use light humor where appropriate.

Remain professional and credible.

Humor must support the sales message.

Do not make the email sound like a comedy script.
"""


instructions3 = intro + """

Your style is concise, direct and executive.

Keep the message short.

Focus on:

- Prospect problem
- ComplAI value
- Clear reason to respond

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
# EMAIL CHECKER
#
# IMPORTANT:
# No tools are attached to this agent.
# Therefore structured output is safe.
# ============================================================

checker = Agent(
    name="Email Checker",

    instructions="""
You review cold sales emails.

Check whether the email:

1. Is professional and appropriate.
2. Contains placeholders such as:

[Name]
[First Name]
[Company]
[Phone]
[Website]
[Your Name]
[Your Website]

3. Count the number of sentences in the
email body, excluding greeting and signature.

Return the result using EmailReview.
""",

    model=model,

    output_type=EmailReview,
)


# ============================================================
# OUTPUT GUARDRAIL
# ============================================================

@output_guardrail
async def email_guardrail(
    ctx,
    agent,
    output,
):

    # Output is SalesEmail
    email_json = output.model_dump_json()

    # Ask checker agent to review it
    result = await Runner.run(
        checker,
        email_json,
    )

    review = result.final_output

    is_problem = (
        review.contains_placeholders
        or not review.is_professional
    )

    return GuardrailFunctionOutput(
        output_info=review,
        tripwire_triggered=is_problem,
    )


# ============================================================
# EMAIL SENDING
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
        timeout=30,
    ) as server:

        server.starttls()

        server.login(
            EMAIL_ADDRESS,
            EMAIL_APP_PASSWORD,
        )

        server.send_message(msg)


# ============================================================
# SEND EMAIL TOOL
# ============================================================

@function_tool
async def send_email_tool(
    subject: str,
    text_body: str,
    html_body: str,
) -> str:

    """
    Send the selected sales email to all
    configured recipients using SMTP.
    """

    await asyncio.to_thread(
        send_email,
        subject,
        text_body,
        html_body,
    )

    return (
        f"Email sent successfully to "
        f"{len(RECIPIENTS)} recipient(s)."
    )


# ============================================================
# SALES AGENTS → TOOLS
# ============================================================

description = """
Use this tool to write a cold sales email.

The agent already knows:

- ComplAI's business
- Sender information
- Its assigned writing style

Generate one complete sales email.
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
# SALES MANAGER
#
# IMPORTANT:
# NO output_type here.
#
# Why?
# Because this agent has tools.
#
# Groq currently rejects JSON/structured output
# combined with function calling.
# ============================================================

manager_instructions = """

You are a Sales Manager at ComplAI.

Your job is to create and send the best
cold sales email.

Follow this workflow exactly.

--------------------------------------------------
STEP 1 — GENERATE THREE DRAFTS
--------------------------------------------------

Use ALL THREE tools:

sales_email_writer_1
sales_email_writer_2
sales_email_writer_3

Generate one draft from each.


--------------------------------------------------
STEP 2 — EVALUATE
--------------------------------------------------

Compare the drafts based on:

- Clarity
- Professionalism
- Persuasiveness
- Relevance
- Natural tone
- Conciseness
- Call to action


--------------------------------------------------
STEP 3 — SELECT
--------------------------------------------------

Select exactly ONE draft.

Do not send multiple drafts.


--------------------------------------------------
STEP 4 — VALIDATE
--------------------------------------------------

Before sending verify:

- No placeholders
- Correct sender information
- Valid greeting
- Professional tone
- Subject exists
- Plain-text body exists
- HTML body exists


--------------------------------------------------
STEP 5 — SEND
--------------------------------------------------

Call send_email_tool.

Send exactly ONE email.

Never send the three drafts separately.


--------------------------------------------------
FINAL RESPONSE
--------------------------------------------------

After sending, return the selected email
as plain text using this format:

SUBJECT:
<subject>

TEXT BODY:
<plain text body>

HTML BODY:
<html body>

Do not call send_email_tool more than once.
"""


sales_manager = Agent(
    name="Sales Manager",

    instructions=manager_instructions,

    tools=[
        tool1,
        tool2,
        tool3,
        send_email_tool,
    ],

    model=model,
)


# ============================================================
# STRUCTURED OUTPUT FORMATTER
#
# This agent has NO tools.
#
# Therefore it can safely use structured output.
# ============================================================

email_formatter = Agent(
    name="Email Formatter",

    instructions="""
You convert the selected sales email into
the required structured format.

Extract:

1. Subject
2. Plain-text body
3. HTML body

Return ONLY the SalesEmail structured output.

Do not modify the content unnecessarily.

Do not invent information.
Do not add placeholders.
""",

    model=model,

    output_type=SalesEmail,

    output_guardrails=[
        email_guardrail,
    ],
)


# ============================================================
# TASK
# ============================================================

task = """

Create a cold sales email for ComplAI.

Use ALL THREE sales email writer tools.

Generate three different versions.

Evaluate all three.

Select exactly ONE.

Validate the selected email.

Use send_email_tool to send exactly ONE email.

After the email is successfully sent,
return the selected email using the required
SUBJECT / TEXT BODY / HTML BODY format.
"""


# ============================================================
# MAIN
# ============================================================

async def main():

    try:

        with trace("Sales Manager"):

            # ----------------------------------------------
            # STAGE 1
            # Manager performs tool orchestration
            # ----------------------------------------------

            manager_result = await Runner.run(
                sales_manager,
                task,
            )

            manager_output = manager_result.final_output


            print("\n" + "=" * 60)
            print("MANAGER COMPLETED")
            print("=" * 60)

            print(manager_output)


            # ----------------------------------------------
            # STAGE 2
            # Convert final output into structured output
            # ----------------------------------------------

            formatter_result = await Runner.run(
                email_formatter,
                manager_output,
            )


            # ----------------------------------------------
            # STAGE 3
            # Structured SalesEmail
            # ----------------------------------------------

            final_email = formatter_result.final_output


            print("\n" + "=" * 60)
            print("STRUCTURED OUTPUT")
            print("=" * 60)

            print(
                json.dumps(
                    final_email.model_dump(),
                    indent=2,
                )
            )


    except OutputGuardrailTripwireTriggered:

        print("\n" + "=" * 60)
        print("OUTPUT GUARDRAIL TRIGGERED")
        print("=" * 60)

        print(
            "The selected email failed the "
            "email quality validation."
        )

        print(
            "The structured output was rejected."
        )


    except Exception as e:

        print("\n" + "=" * 60)
        print("ERROR")
        print("=" * 60)

        print(type(e).__name__)
        print(str(e))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())