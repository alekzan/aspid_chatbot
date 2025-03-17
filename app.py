import sqlite3
import sys
from datetime import datetime
from flask import Flask, request, g, jsonify
import requests
from dotenv import load_dotenv
import os
import redis
import uuid
import logging
from logging.handlers import RotatingFileHandler

# Existing import for your LLM logic
from chatbot_graph import call_model

# Existing import for your LLM logic
from restaurant_graph import (
    call_model_restaurant_bot,
    call_model_as_ai,
    call_model_from_messenger,
)

# NEW: Import the transcription function
from utilities_whatsapp import transcribe_audio_from_whatsapp

# March 14th, 2025
from celery import Celery


# Ensure the database directory exists
os.makedirs("data", exist_ok=True)

DATABASE_PATH = "data/whatsapp_crm.db"


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    return conn


def create_messages_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number_id TEXT,
            telefonoCliente TEXT,
            sender TEXT,
            profile_name TEXT,
            message_type TEXT,
            content TEXT,
            media_id TEXT,
            mime_type TEXT,
            sha256 TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    conn.close()


create_messages_table()

app = Flask(__name__)

file_handler = RotatingFileHandler("flask-app.log", maxBytes=100000, backupCount=10)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(logging.INFO)

load_dotenv(dotenv_path="/var/www/airregio-llm/.env")

VERSION = os.getenv("VERSION")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
RECIPIENT_WAID = os.getenv("RECIPIENT_WAID")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
PHONE_NUMBER_ID_2 = os.getenv("PHONE_NUMBER_ID_2")

# Messenger & Instagram credentials
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
PAGE_ID = os.getenv("PAGE_ID")

# Redis connection details
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0
)


def reset_thread_id(user_key):
    thread_id_number = str(uuid.uuid4())
    redis_client.hset(user_key, "thread_id", thread_id_number)
    return {"configurable": {"thread_id": thread_id_number}}


def get_config(user_key):
    thread_id_number = redis_client.hget(user_key, "thread_id")
    if thread_id_number is None:
        return reset_thread_id(user_key)
    else:
        return {"configurable": {"thread_id": thread_id_number.decode()}}


# Configure Celery after Redis client setup
def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config["CELERY_BROKER_URL"],
        backend=app.config["CELERY_RESULT_BACKEND"],
        include=["app"],  # Explicitly tell Celery where to find tasks
    )
    celery.conf.update(
        task_serializer="pickle",
        accept_content=["pickle"],
        result_serializer="pickle",
        timezone="America/Mexico_City",
        enable_utc=True,
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


# Build Redis URL based on environment
redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
if REDIS_PASSWORD:
    redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

# Configure Flask app
app.config.update(CELERY_BROKER_URL=redis_url, CELERY_RESULT_BACKEND=redis_url)

# Initialize Celery after app is fully configured
celery = make_celery(app)


# Add to app.py
@celery.task(name="app.send_scheduled_review_prompt")  # Add explicit name
def send_scheduled_review_prompt(recipient_phone, phone_number_id):
    scheduled_message = (
        "Este es un ejemplo de cómo AutoFlujo ayuda a los restaurantes a obtener más reseñas en Google Maps 📍.\n\n"
        "📢 Así es como tu cliente recibiría este mensaje después de su visita:\n\n"
        '"¡Hola! Esperamos que hayas disfrutado tu visita a La Cuchara Mágica. Si te gustó la experiencia, ¿nos dejas una reseña de 5 estrellas? '
        'Solo toma un minuto y nos ayuda mucho. ⭐⭐⭐⭐⭐ Visita este link"\n\n'
        "Con AutoFlujo, este proceso es automático, ayudándote a mejorar tu reputación sin esfuerzo. 🚀\n\n"
        "¿Te gustaría implementarlo en tu restaurante? Mándame mensaje aquí: https://bit.ly/4bza3FS"
    )

    try:
        # Get or create config for this user
        user_key = f"whatsapp_conversation_{recipient_phone}"
        config = get_config(user_key)

        # Call the graph as the AI
        call_model_as_ai(recipient_phone, config, scheduled_message)

        # Send the actual WhatsApp message
        send_whatsapp_message(
            recipient=recipient_phone,
            message=scheduled_message,
            phone_number_id=phone_number_id,
            message_type="text",
        )

        # Save to database
        outgoing_message_data = {
            "profile_name": "Chatbot",
            "type": "text",
            "content": scheduled_message,
            "media_id": None,
            "mime_type": None,
            "sha256": None,
        }
        save_message_to_db(
            phone_number_id,
            recipient_phone,
            outgoing_message_data,
            "chatbot",
        )

    except Exception as e:
        logger.error(f"Failed to send scheduled message: {str(e)}")


@app.before_request
def before_request():
    g.conversations = {}


def save_message_to_db(phone_number_id, telefonoCliente, message_data, sender):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO messages (
            phone_number_id, telefonoCliente, sender, profile_name, 
            message_type, content, media_id, mime_type, sha256, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            phone_number_id,
            telefonoCliente,
            sender,
            message_data.get("profile_name", "Unknown"),
            message_data["type"],
            message_data["content"],
            message_data.get("media_id"),
            message_data.get("mime_type"),
            message_data.get("sha256"),
            timestamp,
        ),
    )
    print(f"Saved message: {message_data} from {sender}")
    conn.commit()
    conn.close()


def remove_prefix(number):
    # If the number starts with '521', transform it to '52'
    str_number = str(number)
    if str_number.startswith("521"):
        return "52" + str_number[3:]
    return str_number


def send_whatsapp_message(
    recipient, message, phone_number_id, message_type="text", media_url=None
):
    url = f"https://graph.facebook.com/{VERSION}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    if message_type == "text":
        data = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "text",
            "text": {"body": message},
        }
    elif message_type == "interactive":
        data = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "interactive",
            "interactive": message,
        }
    elif message_type == "image":
        if not media_url:
            raise ValueError("media_url is required for sending image messages.")
        data = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "image",
            "image": {
                "link": media_url,
                "caption": message or "",
            },
        }
    else:
        raise ValueError("Invalid message type. Use 'text', 'interactive', or 'image'.")

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(
            f"Error sending WhatsApp message: {response.status_code}, {response.text}"
        )
    else:
        logger.info(f"WhatsApp message sent to {recipient}")


def send_message_to_platform(platform, recipient_psid, text):
    # Not changing anything for Messenger/IG
    if platform == "instagram":
        url = f"https://graph.facebook.com/v21.0/{PAGE_ID}/messages"
    elif platform == "messenger":
        url = f"https://graph.facebook.com/v17.0/{PAGE_ID}/messages"
    else:
        raise ValueError("Invalid platform. Must be 'messenger' or 'instagram'.")

    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"id": recipient_psid},
        "message": {"text": text},
        "messaging_type": "RESPONSE",
    }
    params = {"access_token": PAGE_ACCESS_TOKEN}

    response = requests.post(url, headers=headers, params=params, json=payload)
    if response.status_code != 200:
        app.logger.error(
            f"Failed to send message to {platform}. Status: {response.status_code}, Response: {response.text}"
        )
    else:
        app.logger.info(f"Message sent successfully to {recipient_psid} on {platform}")


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/status", methods=["GET"])
def status():
    app.logger.info("Status endpoint was reached")
    return jsonify({"status": "running"}), 200


@app.route("/webhook", methods=["POST", "GET"])
@app.route("/webhook/", methods=["POST", "GET"])
def webhook():
    if request.method == "GET":
        if request.args.get("hub.verify_token") == "autoflujo9a":
            return request.args.get("hub.challenge")
        else:
            return "Error de autentificacion."

    data = request.get_json()
    app.logger.info(f"Webhook received data: {data}")

    # WHATSAPP
    if data.get("object") == "whatsapp_business_account":
        try:
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if "value" in change and "messages" in change["value"]:
                        for message in change["value"]["messages"]:
                            telefonoCliente = message["from"]
                            phone_number_id = change["value"]["metadata"][
                                "phone_number_id"
                            ]
                            profile_name = (
                                change["value"]["contacts"][0]
                                .get("profile", {})
                                .get("name", "Unknown")
                            )
                            sender = "user"
                            message_type = message["type"]

                            # NEW: We'll extract content based on type
                            if message_type == "text":
                                content = message["text"]["body"]
                            elif message_type == "interactive":
                                if "list_reply" in message["interactive"]:
                                    list_reply = message["interactive"]["list_reply"]
                                    content = (
                                        f"List Reply ID: {list_reply['id']}, "
                                        f"Title: {list_reply['title']}, "
                                        f"Description: {list_reply['description']}"
                                    )
                                    logger.info(f"User selected list reply: {content}")
                                else:
                                    logger.info(
                                        f"Unsupported interactive type: {message_type}"
                                    )
                                    continue
                            elif message_type == "audio":
                                # If the user sends an audio message
                                media_id = message["audio"]["id"]
                                mime_type = message["audio"]["mime_type"]
                                sha256 = message["audio"]["sha256"]

                                # Transcribe it using our utility
                                content = transcribe_audio_from_whatsapp(
                                    media_id, mime_type, sha256
                                )
                                logger.info(f"Transcribed audio to text: {content}")

                            else:
                                logger.info(
                                    f"Unsupported WhatsApp message type: {message_type}"
                                )
                                continue

                            # Save the incoming message (text or transcribed text) to DB
                            incoming_message_data = {
                                "profile_name": profile_name,
                                "type": message_type,
                                "content": content,
                                "media_id": (
                                    message["audio"]["id"]
                                    if message_type == "audio"
                                    else None
                                ),
                                "mime_type": (
                                    message["audio"]["mime_type"]
                                    if message_type == "audio"
                                    else None
                                ),
                                "sha256": (
                                    message["audio"]["sha256"]
                                    if message_type == "audio"
                                    else None
                                ),
                            }
                            save_message_to_db(
                                phone_number_id,
                                telefonoCliente,
                                incoming_message_data,
                                sender,
                            )

                            if phone_number_id == PHONE_NUMBER_ID:
                                # Process the text with call_model
                                user_key = f"whatsapp_conversation_{telefonoCliente}"
                                g.config = get_config(user_key)
                                client_phone = remove_prefix(telefonoCliente)
                                # response = call_model(content, client_phone, g.config)
                                response, message_type = call_model(
                                    content, client_phone, g.config
                                )

                                # Send the response back as a text
                                if message_type == "text":
                                    send_whatsapp_message(
                                        client_phone,
                                        response,
                                        phone_number_id,
                                        message_type,
                                    )
                                elif message_type == "image":
                                    send_whatsapp_message(
                                        client_phone,
                                        response,
                                        phone_number_id,
                                        message_type,
                                        media_url="https://i.ibb.co/cvBV385/assy-aspid.png",
                                    )
                                # Save outgoing response
                                outgoing_message_data = {
                                    "profile_name": "Chatbot",
                                    "type": "text",  # Always "text" for now
                                    "content": response,
                                    "media_id": None,
                                    "mime_type": None,
                                    "sha256": None,
                                }
                                save_message_to_db(
                                    phone_number_id,
                                    telefonoCliente,
                                    outgoing_message_data,
                                    "chatbot",
                                )
                            elif (
                                phone_number_id == PHONE_NUMBER_ID_2
                            ):  # production: "289382677601457":
                                # Process the text with call_model
                                user_key = f"whatsapp_conversation_{telefonoCliente}"
                                g.config = get_config(user_key)
                                client_phone = remove_prefix(telefonoCliente)

                                # Check if this is first interaction
                                first_msg_key = f"first_interaction_{client_phone}"
                                if not redis_client.exists(first_msg_key):
                                    # Schedule the message and set lock
                                    redis_client.set(
                                        first_msg_key, "scheduled", ex=86400
                                    )  # 24h expiration
                                    send_scheduled_review_prompt.apply_async(
                                        args=[client_phone, phone_number_id],
                                        countdown=1800,  # 30 minutes
                                        queue="default",  # Add explicit queue name
                                    )
                                # Call the model
                                response = call_model_restaurant_bot(
                                    content, client_phone, g.config
                                )

                                # Send the response back as a text
                                send_whatsapp_message(
                                    client_phone,
                                    response,
                                    phone_number_id,
                                    message_type="text",
                                )

                                # Save outgoing response
                                outgoing_message_data = {
                                    "profile_name": "Chatbot",
                                    "type": "text",  # Always "text" for now
                                    "content": response,
                                    "media_id": None,
                                    "mime_type": None,
                                    "sha256": None,
                                }
                                save_message_to_db(
                                    phone_number_id,
                                    telefonoCliente,
                                    outgoing_message_data,
                                    "chatbot",
                                )
        except KeyError as e:
            logger.error(f"Error processing WhatsApp webhook: {e}")

    return "OK", 200


if __name__ != "__main__":
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
