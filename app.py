import os
import sys
import uuid
import sqlite3
import logging
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, request, g, jsonify
from logging.handlers import RotatingFileHandler

# -------------------------------------------------------------------
# Carga de variables de entorno
# -------------------------------------------------------------------
load_dotenv(dotenv_path="/var/www/airregio-llm/.env")

VERSION = os.getenv("VERSION")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
RECIPIENT_WAID = os.getenv("RECIPIENT_WAID")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
PHONE_NUMBER_ID_2 = os.getenv("PHONE_NUMBER_ID_2")

# Credenciales de Messenger e Instagram
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
PAGE_ID = os.getenv("PAGE_ID")

# -------------------------------------------------------------------
# Funciones y modelos externos
# -------------------------------------------------------------------
from chatbot_graph import call_model
from restaurant_graph import (
    call_model_restaurant_bot,
    call_model_as_ai,
    call_model_from_messenger,
)
from utilities_whatsapp import transcribe_audio_from_whatsapp

# -------------------------------------------------------------------
# Configuración de base de datos
# -------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
DATABASE_PATH = "data/whatsapp_crm.db"


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    return conn


def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Tabla de mensajes
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

    # Tabla para guardar thread_id por usuario
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_configs (
            user_key TEXT PRIMARY KEY,
            thread_id TEXT
        )
        """
    )
    conn.commit()
    conn.close()


create_tables()

# -------------------------------------------------------------------
# Configuración de Flask y logging
# -------------------------------------------------------------------
app = Flask(__name__)

file_handler = RotatingFileHandler("flask-app.log", maxBytes=100_000, backupCount=10)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Utilidades de conversación (sin Redis)
# -------------------------------------------------------------------
def reset_thread_id(user_key: str):
    thread_id_number = str(uuid.uuid4())
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO conversation_configs (user_key, thread_id)
        VALUES (?, ?)
        """,
        (user_key, thread_id_number),
    )
    conn.commit()
    conn.close()
    return {"configurable": {"thread_id": thread_id_number}}


def get_config(user_key: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT thread_id FROM conversation_configs WHERE user_key = ?
        """,
        (user_key,),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return reset_thread_id(user_key)
    return {"configurable": {"thread_id": row[0]}}


# -------------------------------------------------------------------
# Helpers varios
# -------------------------------------------------------------------
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
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    conn.commit()
    conn.close()


def remove_prefix(number: str) -> str:
    if number.startswith("521"):
        return "52" + number[3:]
    return number


def send_whatsapp_message(
    recipient: str,
    message: str,
    phone_number_id: str,
    message_type: str = "text",
    media_url: str | None = None,
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
            raise ValueError("media_url is required for image messages.")
        data = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "image",
            "image": {"link": media_url, "caption": message or ""},
        }
    else:
        raise ValueError("Invalid message type.")

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        logger.error(
            f"Error sending WhatsApp message: {response.status_code}, {response.text}"
        )
    else:
        logger.info(f"WhatsApp message sent to {recipient}")


def send_message_to_platform(platform: str, recipient_psid: str, text: str):
    if platform == "instagram":
        url = f"https://graph.facebook.com/v21.0/{PAGE_ID}/messages"
    elif platform == "messenger":
        url = f"https://graph.facebook.com/v17.0/{PAGE_ID}/messages"
    else:
        raise ValueError("Invalid platform.")

    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"id": recipient_psid},
        "message": {"text": text},
        "messaging_type": "RESPONSE",
    }
    params = {"access_token": PAGE_ACCESS_TOKEN}

    response = requests.post(url, headers=headers, params=params, json=payload)
    if response.status_code != 200:
        logger.error(
            f"Failed to send message to {platform}. "
            f"Status: {response.status_code}, Response: {response.text}"
        )
    else:
        logger.info(f"Message sent successfully to {recipient_psid} on {platform}")


# -------------------------------------------------------------------
# Rutas básicas
# -------------------------------------------------------------------
@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/status", methods=["GET"])
def status():
    logger.info("Status endpoint reached")
    return jsonify({"status": "running"}), 200


# -------------------------------------------------------------------
# Webhook principal
# -------------------------------------------------------------------
@app.route("/webhook", methods=["POST", "GET"])
@app.route("/webhook/", methods=["POST", "GET"])
def webhook():
    if request.method == "GET":
        if request.args.get("hub.verify_token") == "autoflujo9a":
            return request.args.get("hub.challenge")
        return "Error de autentificacion."

    data = request.get_json()
    logger.info(f"Webhook received data: {data}")

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

                            # --------------------------------------------------
                            # Lectura de contenido según tipo
                            # --------------------------------------------------
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
                                media_id = message["audio"]["id"]
                                mime_type = message["audio"]["mime_type"]
                                sha256 = message["audio"]["sha256"]
                                content = transcribe_audio_from_whatsapp(
                                    media_id, mime_type, sha256
                                )
                                logger.info(f"Transcribed audio to text: {content}")
                            else:
                                logger.info(
                                    f"Unsupported WhatsApp message type: {message_type}"
                                )
                                continue

                            # --------------------------------------------------
                            # Guarda mensaje entrante
                            # --------------------------------------------------
                            incoming_message_data = {
                                "profile_name": profile_name,
                                "type": message_type,
                                "content": content,
                                "media_id": message["audio"]["id"]
                                if message_type == "audio"
                                else None,
                                "mime_type": message["audio"]["mime_type"]
                                if message_type == "audio"
                                else None,
                                "sha256": message["audio"]["sha256"]
                                if message_type == "audio"
                                else None,
                            }
                            save_message_to_db(
                                phone_number_id,
                                telefonoCliente,
                                incoming_message_data,
                                sender,
                            )

                            # --------------------------------------------------
                            # Procesa y responde (dos números posibles)
                            # --------------------------------------------------
                            user_key = f"whatsapp_conversation_{telefonoCliente}"
                            g.config = get_config(user_key)
                            client_phone = remove_prefix(telefonoCliente)

                            if phone_number_id == PHONE_NUMBER_ID:
                                response, out_type = call_model(
                                    content, client_phone, g.config
                                )

                                if out_type == "image":
                                    send_whatsapp_message(
                                        client_phone,
                                        response,
                                        phone_number_id,
                                        message_type="image",
                                        media_url="https://i.ibb.co/cvBV385/assy-aspid.png",
                                    )
                                else:
                                    send_whatsapp_message(
                                        client_phone,
                                        response,
                                        phone_number_id,
                                        message_type="text",
                                    )

                            elif phone_number_id == PHONE_NUMBER_ID_2:
                                response = call_model_restaurant_bot(
                                    content, client_phone, g.config
                                )
                                send_whatsapp_message(
                                    client_phone,
                                    response,
                                    phone_number_id,
                                    message_type="text",
                                )

                            # --------------------------------------------------
                            # Guarda mensaje saliente
                            # --------------------------------------------------
                            outgoing_message_data = {
                                "profile_name": "Chatbot",
                                "type": "text",
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


# -------------------------------------------------------------------
# Arranque
# -------------------------------------------------------------------
if __name__ != "__main__":
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
