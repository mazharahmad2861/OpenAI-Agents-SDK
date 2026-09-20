import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI

load_dotenv()


# -----------------------------
# Gmail configuration
# -----------------------------

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
RECEIVER_EMAIL_ADDRESS = os.getenv("RECEIVER_EMAIL_ADDRESS")

# -----------------------------
# Email tool
# -----------------------------

@function_tool
def send_email(
    to: str,
    subject: str,
    text_body: str,
) -> str:
    """Send an email using Gmail SMTP."""

    try:
        msg = EmailMessage()

        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to
        msg["Subject"] = subject

        msg.set_content(text_body)

        with smtplib.SMTP(EMAIL_SMTP_SERVER, 587) as server:
            server.starttls()

            server.login(
                EMAIL_ADDRESS,
                EMAIL_APP_PASSWORD
            )

            server.send_message(msg)

        return f"Email sent successfully to {to}"

    except Exception as e:
        return f"Failed to send email: {e}"


# -----------------------------
# Groq client
# -----------------------------

groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Groq model
model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=groq_client,
)


# -----------------------------
# Mail Agent
# -----------------------------

mail_agent = Agent(
    name="Mail Agent",

    instructions="""
    You are a helpful mail agent.

    Your job is to send emails when the user asks.

    When the user provides:
    - recipient
    - subject
    - message

    use the send_email tool.

    Do not claim that an email was sent unless
    the send_email tool successfully reports it.
    """,

    model=model,

    tools=[
        send_email
    ]
)


# -----------------------------
# Run agent
# -----------------------------

result = Runner.run_sync(
    mail_agent,
    """
    Send an email to alexmohammad1815@gmail.com

    Subject: Meeting Reminder

    Message:
    Hi, this is a reminder that our meeting
    is scheduled for tomorrow at 5 PM.
    """
)

print(result.final_output)