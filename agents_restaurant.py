# agents.py
# Standard library imports
import os
import os.path
from datetime import datetime

# Third-party library imports
from dotenv import load_dotenv
from typing import Optional, Dict
import pytz

# LangChain and related imports
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_pinecone import PineconeVectorStore

# Pinecone and Airtable imports
from pinecone import Pinecone
from pyairtable import Api

load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Restaurante Bot Tests"


os.environ["AIRTABLE_API_KEY"] = os.getenv("AIRTABLE_API_KEY")

gpt = "gpt-4o-mini"

llama_3_1 = "llama-3.1-8b-instant"
llama_3_2 = "llama-3.2-90b-vision-preview"
llama_3_3 = "llama-3.3-70b-versatile"

# llm = ChatOpenAI(model=gpt, temperature=0.2)
llm = ChatGroq(model=llama_3_1, temperature=0.2)
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("chatbot-restaurante")

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize Retriever
retriever_general = vector_store.as_retriever(
    search_kwargs={"k": 3, "filter": {"source": "faqs"}}
)
retriever_menu = vector_store.as_retriever(
    search_kwargs={"k": 4, "filter": {"source": "menu"}}
)

# Create tool
general_retriever_tool = create_retriever_tool(
    retriever_general,
    "general_retriever_tool",
    "Search and return general information about Restaurant FAQs.",
)
menu_retriever_tool = create_retriever_tool(
    retriever_menu,
    "menu_retriever_tool",
    "Search and return information about Restaurant Menu.",
)

# Hardcoded Airtable configuration
DEFAULT_BASE_ID = "appWZExxj1q0LD4n1"
DEFAULT_TABLE_NAME = "tbll5UzqzJG0f2YMJ"


def combine_date_and_time(
    date_str: str, time_str: str = "", timezone="America/Mexico_City"
) -> str:
    """
    Combines a date string (YYYY-MM-DD) and a time string (HH:MM in 24-hour format)
    into an ISO 8601 datetime adjusted to UTC.

    Args:
    date_str (str): Date in 'YYYY-MM-DD' format.
    time_str (str): Time in 'HH:MM' format. If empty, defaults to '00:00'.
    timezone (str): The local timezone to adjust from. Defaults to 'America/Mexico_City'.

    Returns:
    str: ISO 8601 formatted datetime string in UTC (e.g., '2024-12-02T20:30:00Z').
    """
    if not date_str:
        raise ValueError("Date string is required.")

    # Set default time if not provided
    time_str = time_str or "00:00"

    # Combine date and time into a single naive datetime
    naive_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

    # Localize to the specified timezone
    local_tz = pytz.timezone(timezone)
    local_datetime = local_tz.localize(naive_datetime)

    # Convert to UTC
    utc_datetime = local_datetime.astimezone(pytz.utc)

    # Return in ISO 8601 format
    return utc_datetime.isoformat()


def add_user_to_restaurant_db(
    nombre: str,
    telefono: str,
    email: str,
    fecha: str,
    hora: str,
    numero_personas: int,
    notes: str = "",
):
    """
    Adds a record to the Restaurant AirTable table with the given details.

    Args:
        nombre: Full name of the person.
        telefono: Phone number of the person.
        email: Email of the person.
        fecha: Date of the reservation in YYYY-MM-DD format.
        hora: Time of the reservation in HH:MM (24-hour format).
        numero_personas: Number of people for the reservation.
        notes: Additional notes for the reservation. Defaults to an empty string.

    Returns:
        dict: The added record, or an error message if something fails.
    """
    try:
        # Ensure the API key is available
        api_key = os.getenv("AIRTABLE_API_KEY")
        if not api_key:
            raise ValueError(
                "AIRTABLE_API_KEY is not set in the environment variables."
            )

        # Initialize the Airtable API
        api = Api(api_key)

        # Access the table using the API instance
        table = api.table(DEFAULT_BASE_ID, DEFAULT_TABLE_NAME)

        # Combine date and time into ISO 8601 format
        fecha_y_hora = combine_date_and_time(fecha, hora)

        # Create the record data
        new_record_data = {
            "Nombre": nombre,
            "Teléfono": telefono,
            "Email": email,
            "Fecha y Hora": fecha_y_hora,
            "Nº Personas": numero_personas,
            "Estatus": "Recibida",  # Hardcoded status
            "Notes": notes,
        }

        # Add the record to the table
        return {"success": True, "record": table.create(new_record_data)}

    except Exception as e:
        # Return the error message in case of failure
        return {"success": False, "error": str(e)}


def update_reservation_in_restaurant_db(
    record_id: str,
    nombre: Optional[str] = None,
    telefono: Optional[str] = None,
    email: Optional[str] = None,
    fecha: Optional[str] = None,
    hora: Optional[str] = None,
    numero_personas: Optional[int] = None,
    notes: Optional[str] = None,
):
    """
    Updates a record in the Restaurant AirTable table with the given details.

    Args:
        record_id: The ID of the record to update.
        nombre: Updated full name of the person (optional).
        telefono: Updated phone number of the person (optional).
        email: Updated email of the person (optional).
        fecha: Updated date of the reservation in YYYY-MM-DD format (optional).
        hora: Updated time of the reservation in HH:MM (24-hour format) (optional).
        numero_personas: Updated number of people for the reservation (optional).
        notes: Updated additional notes for the reservation (optional).

    Returns:
        dict: The updated record, or an error message if something fails.
    """
    try:
        # Ensure the API key is available
        api_key = os.getenv("AIRTABLE_API_KEY")
        if not api_key:
            raise ValueError(
                "AIRTABLE_API_KEY is not set in the environment variables."
            )

        # Initialize the Airtable API
        api = Api(api_key)

        # Access the table using the API instance
        table = api.table(DEFAULT_BASE_ID, DEFAULT_TABLE_NAME)

        # Combine date and time into ISO 8601 format if both are provided
        fecha_y_hora = combine_date_and_time(fecha, hora) if fecha and hora else None

        # Prepare fields to update
        updated_fields = {}
        if nombre is not None:
            updated_fields["Nombre"] = nombre
        if telefono is not None:
            updated_fields["Teléfono"] = telefono
        if email is not None:
            updated_fields["Email"] = email
        if fecha_y_hora is not None:
            updated_fields["Fecha y Hora"] = fecha_y_hora
        if numero_personas is not None:
            updated_fields["Nº Personas"] = numero_personas
        if notes is not None:
            updated_fields["Notes"] = notes

        # Ensure there is something to update
        if not updated_fields:
            raise ValueError("No fields to update were provided.")

        # Update the record
        updated_record = table.update(record_id, updated_fields)
        return {"success": True, "record": updated_record}

    except Exception as e:
        # Return the error message in case of failure
        return {"success": False, "error": str(e)}


def cancel_reservation_in_restaurant_db(
    record_id: str,
    notes: Optional[str] = None,
):
    """
    Cancels a reservation in the Restaurant AirTable table by updating the 'estatus' to 'Cancelada'.
    Optionally updates the notes for the reservation.

    Args:
        record_id: The ID of the record to cancel.
        notes: Additional notes for the cancellation (optional).

    Returns:
        dict: The updated record, or an error message if something fails.
    """
    try:
        # Ensure the API key is available
        api_key = os.getenv("AIRTABLE_API_KEY")
        if not api_key:
            raise ValueError(
                "AIRTABLE_API_KEY is not set in the environment variables."
            )

        # Initialize the Airtable API
        api = Api(api_key)

        # Access the table using the API instance
        table = api.table(DEFAULT_BASE_ID, DEFAULT_TABLE_NAME)

        # Prepare fields to update
        updated_fields = {"Estatus": "Cancelada"}
        if notes is not None:
            updated_fields["Notes"] = notes

        # Update the record
        updated_record = table.update(record_id, updated_fields)
        return {"success": True, "record": updated_record}

    except Exception as e:
        # Return the error message in case of failure
        return {"success": False, "error": str(e)}


from typing import Optional, Dict


def recordar_informacion_importante(
    nombre_del_cliente: Optional[str] = None,
    telefono: Optional[str] = None,
    correo_electronico: Optional[str] = None,
    numero_de_personas: Optional[int] = None,
    fecha: Optional[str] = None,
    hora: Optional[str] = None,
    solicitudes_extra: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Agrega la información del usuario a tu base de datos.

    Args:
        nombre_del_cliente: Nombre del cliente.
        telefono: Número de teléfono del cliente.
        correo_electronico: Correo electrónico del cliente.
        numero_de_personas: Número de personas (deja este campo vacío o como 'null' si no se menciona).
        fecha: Fecha (especificar YYYY-MM-DD si está presente, o interpretar lo que el mensaje indique, como "este viernes").
        hora: Hora (Especificar en formato 24 horas. Incluye este campo únicamente si la hora exacta está mencionada en el mensaje de forma explícita.
              Si no se menciona una hora específica, deja este campo vacío.).
        solicitudes_extra: Solicitudes específicas sobre la reservación. Solamente referentes a la reservación en el restaurante.
                           No agregues notas sobre la conversación.

    Returns:
        Diccionario con los valores procesados de la reservación. Las claves son:
        - "name": nombre_del_cliente.
        - "phone": telefono.
        - "email": correo_electronico.
        - "persons_number": numero_de_personas.
        - "date": fecha.
        - "time": hora.
        - "requests": solicitudes_extra.
    """
    print("TOOL: recordar_informacion_importante")
    return {
        "name": nombre_del_cliente or None,
        "phone": telefono or None,
        "email": correo_electronico or None,
        "persons_number": (
            numero_de_personas if numero_de_personas is not None else None
        ),
        "date": fecha or None,
        "time": hora or None,
        "requests": solicitudes_extra or None,
    }


# Example Usage
# result = add_user_to_restaurant_db(
#    nombre="Ana del Valle",
#    telefono="+525555555555",
#    email="ana_v@example.com",
#    fecha="2024-12-08",
#    hora="19:30",
#    numero_personas=5,
#    notes="Evitar mariscos",
# )
#
# if result["success"]:
#    print(f"New Record ID: {result['record']['id']}")
#    print(f"Fields: {result['record']['fields']}")
# else:
#    print(f"Error: {result['error']}")


tools = [
    general_retriever_tool,
    menu_retriever_tool,
    add_user_to_restaurant_db,
    update_reservation_in_restaurant_db,
    cancel_reservation_in_restaurant_db,
]
extract_tools = [recordar_informacion_importante]
# Obtener la fecha y hora actuales
current_datetime = datetime.now().strftime("Hoy es %d de %B de %Y a las %I:%M %p.")

react_prompt = f"""Eres un asistente que trabaja para el restaurante La Cuchara Mágica. 
Tu principal tarea es ayudar al usuario a obtener información sobre nuestro restaurante, menú, reservaciones o cualquier otra consulta que tenga.

Responde de manera concisa. No más de 3 oraciones.

Cuando un usuario quiera realizar una reservación, recopila la siguiente información faltante de manera gradual y amigable:

Información a obtener o ya obtenida:
- Nombre del cliente: {{name}}
- Teléfono: {{phone}} (confirmar con usuario si se te da desde el inicio) Debe contener el código del país como éste "+52" y los 10 dígitos.
- Correo electrónico: {{email}} 
- Número de personas: {{persons_number}}
- Fecha: {{date}}
- Hora: {{time}}
- Alguna solicitud extra (opcional): {{requests}}

NOTA: El teléfono a veces se obtiene automáticamente desde un inicio, pero confírmalo con el usuario.
NOTA: Si no se tienes el teléfono, solícitalo. Recuerda que debe contener el código del país como éste "+52" y los 10 dígitos.

Responde en el mismo idioma en el que el usuario se comunique contigo.  
Asegúrate de mantener la conversación amistosa y clara, añadiendo saltos de línea para que los mensajes sean fáciles de leer.
Always answer based only on the information retrieved with your tools.

Interpreta cualquier información ambigua sobre la fecha y la hora, considerando el siguiente contexto temporal:
{{current_datetime}}
Considera que el lugar está abierto de lunes a domingo de 11:00 a 23:00 horas.

Si el usuario te da información sobre la fecha y hora de reservación, pero no estás seguro, confirma.

Si se te indica ID de la reservación quiere decir que ya está en el sistema por lo que si el usuario quiere hacer cambios a la reservación tendrás que usar la herramienta update_reservation_in_restaurant_db y pasar el ID.

- ID de la reservación: {{id}}
- ¿Ya fue agendado?: {{booked_status}}

## Herramientas disponibles:
- general_retriever_tool: Utiliza esta herramienta para obtener información general sobre el restaurante que te pregunte el usuario.
- menu_retriever_tool: Utiliza esta herramienta para obtener información sobre el menú del restaurante.
- add_user_to_restaurant_db: Utiliza esta herramienta cuando tengas TODOS los datos (nombre, teléfono, email, número de personas, fecha, hora y opcionalmente solicitud extra) para agregar la info a la base de datos. 
- update_reservation_in_restaurant_db: Si ya fue agendada la reservación (True), utiliza esta herramienta para hacer actualizaciones usando el ID de la reservación. Si te piden cambiar la hora tienes que pasar la fecha (YYYY-MM-DD) y hora (HH:MM) en formato 24 horas. NO puedes pasar solamente la hora.
- cancel_reservation_in_restaurant_db: CUIDADO, usa solo si el usuario dice textualmente que quiere cancelar, usa el ID de la reservación. Solo si el cliente te informó por qué cancela, pasa esa inforamción a las Notas.

RECUERDA:  
- Mantén la conversación ligera y profesional, de manera concisa y breve. No más de 3 oraciones.
- El usuario no debe enterarse que la información fue enviada a la base de datos. Solo debe saber la información referente a su reservación.
- Cuando la reservación haya sido hecha correctamente, agradece al usuario y coméntale que se le enviará una confirmación por WhatsApp o correo antes de la reservación.
"""

info_extraction_prompt = f"""
Extrae en Español la siguiente información de este mensaje, utilizada para agendar una mesa o atender la solicitud del cliente en el restaurante La Cuchara Mágica. Usa tu herramienta para almacenar dicha información en tu base de datos:
Información a obtener o ya obtenida:
- Nombre: {{name}}
- Número de teléfono: {{phone}}
- Correo electrónico: {{email}} 
- Número de personas (deja este campo vacío o como 'null' si no se menciona): {{persons_number}}
- Fecha (especificar YYYY-MM-DD si están presentes, o interpretar lo que el mensaje indique, como "este viernes"): {{date}}
- Hora (Especificar en formato 24 horas. Incluye este campo únicamente si la hora exacta está mencionada en el mensaje de forma explícita. Si no se menciona una hora específica, deja este campo vacío.): {{time}}
- Solicitudes específicas sobre la reservación. Solamente referentes a la reservación en el restaurante. No agregues notas sobre la conversación: {{requests}}

NOTA: Solo actualiza la información que ya tengas si es importante. Si primero el usuario dice que se llama Juan Pérez y luego menciona que se llama Juan, NO actualices la información.

## Herramientas disponibles:
- recordar_informacion_importante: Usa esta herramienta cada que el usuario te de información referente a la reservación  como nombre, teléfono, email, número de personas, fecha, hora y solicitud extra. Esto guardará esa información.

If the message mentions a general time of day like 'mañana en la noche' without specifying an exact time, leave the Hora field empty.

IMPORTANTE: NO inventes información que no está explícita. Usa `null` (sin comillas) para cualquier campo que no tenga información disponible.

Por favor, interpreta cualquier información ambigua sobre la fecha y la hora, considerando el siguiente contexto temporal:
{{current_datetime}}

No incluyas ninguna explicación o texto adicional; solo devuelve el objeto JSON.
"""
