import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
RECEIVER_EMAIL_ADDRESS = os.getenv("RECEIVER_EMAIL_ADDRESS")

def send_email(subject, text_body, html_body):
    msg = EmailMessage()

    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECEIVER_EMAIL_ADDRESS
    msg["Subject"] = subject

    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html") #optional

    with smtplib.SMTP(EMAIL_SMTP_SERVER, 587) as server:
        server.starttls()
        server.login(
            EMAIL_ADDRESS,
            EMAIL_APP_PASSWORD
        )
        server.send_message(msg)


send_email(
    "Quote of the Day",
    "Hii, Dream big, work hard, and make it happen.",
    "<html><body><strong>Dream Big</strong> Act Now</body></html>"
)