import operator
import os
import ssl
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool

# from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# from pinecone import Pinecone

import os.path

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Aspid Pro"


# os.environ["AIRTABLE_API_KEY"] = os.getenv("AIRTABLE_API_KEY")

gpt_4o = "gpt-4o-2024-11-20"
gpt = "gpt-4o-mini"

llama_3_1 = "llama-3.1-8b-instant"
llama_3_2 = "llama-3.2-90b-vision-preview"
llama_3_3 = "llama-3.3-70b-versatile"

llm = ChatOpenAI(model=gpt, temperature=0.2)
# llm = ChatGroq(model=llama_3_3, temperature=0.2)

# Carga vector store
# Configuraciones para informacion_de_tienda
base_name_informacion_de_tienda = "informacion_de_tienda"
persist_directory_informacion_de_tienda = (
    f"data/chroma_vectorstore_{base_name_informacion_de_tienda}"
)

vector_store_informacion_de_tienda = Chroma(
    collection_name=base_name_informacion_de_tienda,
    embedding_function=embeddings,
    persist_directory=persist_directory_informacion_de_tienda,
)

# Configuraciones para product_data
base_name_product_data = "product_data"
persist_directory_product_data = f"data/chroma_vectorstore_{base_name_product_data}"

vector_store_product_data = Chroma(
    collection_name=base_name_product_data,
    embedding_function=embeddings,
    persist_directory=persist_directory_product_data,
)

retriever_info_tienda = vector_store_informacion_de_tienda.as_retriever()
retriever_info_productos = vector_store_product_data.as_retriever()

retriever_tool_faq_tienda = create_retriever_tool(
    retriever_info_tienda,
    "retriever_info_tienda",
    "Search and return information about AspidPro shipping, promotions, returns, contact information, ingredients/components/formulas, and skincare routine.",
)

retriever_tool_data_products = create_retriever_tool(
    retriever_info_productos,
    "retriever_info_productos",
    "Retrieve comprehensive information about AspidPro's product range across all categories. Access details such as product codes, skin type compatibility, sizes, prices, and descriptions of key benefits and formulations to assist with product selection and skincare routines.",
)

# This function calls for human help by sending an email.
# It is designed to be passed to an LLM, so the LLM can trigger human assistance
# whenever it cannot handle a request.


@tool
def call_for_human_help(
    client_phone: str, body: str, email_receiver: str = "alejandro_capellan@hotmail.com"
) -> str:
    """
    Call for human assistance when the AI cannot resolve the user's query.

    Args:
        client_phone (str): The phone number of the user who needs human assistance.
        body (str): Explanation or details of what the user requires that the AI cannot address.
        email_receiver (str, optional): The email address where the alert should be sent.
                                        Defaults to "alejandro_capellan@hotmail.com".

    Returns:
        str: A message indicating the result of the attempt to send the email.
    """

    # The subject must follow the format "Un usuario con teléfono {client_phone} necesita tu ayuda"
    subject = f"Un usuario con teléfono {client_phone} necesita tu ayuda"

    # Email body template (in Spanish) to ensure clarity and completeness
    email_body_template = f"""
    El chatbot no pudo contestar una duda del usuario con el teléfono: {client_phone}

    Éste es el reporte del chatbot:
    {body}

    Por favor, revisa esta solicitud y comunícate con el usuario lo antes posible para resolver su problema.
    """

    email_sender = "alejo.capellan@gmail.com"
    email_password = os.environ.get("EMAIL_PASSWORD")

    # Create the email content
    em = EmailMessage()
    em["From"] = email_sender
    em["To"] = email_receiver
    em["Subject"] = subject
    em.set_content(email_body_template)

    # Add SSL (layer of security)
    context = ssl.create_default_context()

    try:
        # Log in and send the email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(email_sender, email_password)
            smtp.sendmail(email_sender, email_receiver, em.as_string())
        return f"He llamado a un asistente humano, se comunicará contigo en breve para ayudarte."
    except Exception as e:
        print(
            f"Fallo al enviar el correo para asistencia humana a {email_receiver}. Error: {e}"
        )
        return f"Fallo al enviar el correo para asistencia humana a {email_receiver}. Error: {e}"


# clasificar_usuario
@tool
def clasificar_usuario(tipo_de_piel: str):
    """
    Classifies the user's skin type based on the provided category.

    Args:
        tipo_de_piel (str): The skin type classification. Must be one of the following values:
            - "Piel seca"
            - "Piel normal"
            - "Piel grasa"
    Returns:
        str: The skin type assigned to the user.
    """
    return f"Tipo de piel: {tipo_de_piel}. Pregunta al usuario si desea una rutina y consejos para su tipo de piel."


# start_skin_test
@tool
def start_skin_test():
    """
    Initiates the skin test process by calling the assistant.

    Returns:
        str: A message indicating that the skin test has been initiated.
    """
    return "Vamos a aplicar un skin test para conocer el tipo de piel del usuario. Pregunta al usuario si está listo para comenzar."


# Add information about user's skin type and if they are a professional or general client
@tool
def clasificar_variables_de_usuario(
    tipo_de_piel: str = None, tipo_de_cliente: str = None
):
    """
    Returns the values passed for tipo_de_piel and tipo_de_cliente without additional text.

    Args:
        tipo_de_piel (str, optional): The skin type classification. Must be one of the following values:
            - "Piel seca"
            - "Piel normal"
            - "Piel grasa"
        tipo_de_cliente (str, optional): The client type classification. Must be one of the following values:
            - "Profesional de Belleza"
            - "Cliente General"

    Returns:
        dict: A dictionary containing the passed values for `tipo_de_piel` and/or `tipo_de_cliente`.
    """
    result = {}

    if tipo_de_piel in ["Piel seca", "Piel normal", "Piel grasa"]:
        result["tipo_de_piel"] = tipo_de_piel

    if tipo_de_cliente in ["Profesional de Belleza", "Cliente General"]:
        result["tipo_de_cliente"] = tipo_de_cliente

    return result


tools = [
    retriever_tool_faq_tienda,
    retriever_tool_data_products,
    call_for_human_help,
    start_skin_test,
    clasificar_variables_de_usuario,
]

tools_for_skin_test = [
    retriever_tool_faq_tienda,
    retriever_tool_data_products,
    call_for_human_help,
    clasificar_usuario,
]

react_prompt = f"""Eres Génesis, un asistente virtual de AspidPro. 

Eres un asistente profesional y amable que trabaja para AspidPro, una farmacéutica especializada en cosmética. 

Responde de manera concisa. No más de 3 oraciones.

Tu primera tarea es preguntar al usuario si es Profesional de Belleza o Cliente General. Explícale que necesitas saber para atenderlo mejor. Si no te contesta tú continúa atendiéndolo como Cliente General. Pero nunca des información reservada para profesionales de la belleza si el usuario no te dice que es un Profesional de la Belleza.

TIPO DE CLIENTE (profesional o cliente general): <{{tipo_de_cliente}}>. (NOTA: Si este valor está vacío es porque el usuario no ha respondido tu primera pregunta sobre si es Profesional de Belleza o Cliente general).

Responde en el mismo idioma en el que el usuario se comunique contigo.  

Si es tu primera interacción con el usuario no olvides saludarlo y presentarte.

Asegúrate de mantener la conversación amistosa y clara, añadiendo saltos de línea para que los mensajes sean fáciles de leer. Pero mantén tu respuesta concisa.

Si el usuario es cliente, enfócate en vender con mensajes concisos; si es profesional, brinda explicaciones técnicas útiles, dale listas de productos y no te bases en su tipo de piel para ofrecerle productos, eso es solo para clientes generales. Usa tus herramientas para buscar la información que te solicite cualquiera de los usuarios.

Si el usuario es Cliente General, antes de recomendar productos, pregunta al usuario su tipo de piel. Si éste no lo sabe usa la herramienta start_skin_test para aplicar un skin test y conocer su tipo de piel. Inmediatamente después de usar esta herramienta coméntale al usuario que van a hacer un pequeño test para conocer su tipo de piel y pregúntale si está listo para comenzar.

Si el usuario no te quiere dar su tipo de piel y no quiere hacer el skin test y solo quiere información sobre los productos, NO hagas el skin test y no insistas. Tus herramientas de retriever son suficientes para bríndale la información que solicita.

Always answer based only on the information retrieved with your tools.

Si no sabes la respuesta di que no tienes información al respecto usa tu herramienta call_for_human_help. Si desconoces la respuesta jamás le recomiendes al usuario contactar directamente a la empresa. Para eso está el call_for_human_help.

Si el skin test ya se hizo, éste es el resultado de la clasificación de la piel del usuario: 

TIPO DE PIEL: <{{tipo_de_piel}}>. (NOTA: Si este valor está vacío es porque el skin test no se ha realizado, ya sea porque el usuario te dijo su tipo de piel o por que es un Profesional de Belleza y no hay necesidad de hacer test).

Interpreta cualquier información ambigua sobre la fecha y la hora, considerando el siguiente contexto temporal:
{{current_datetime}}

El teléfono del usuario es el siguiente: {{client_phone}}

## Herramientas disponibles (Puedes hacer dos llamadas seguidas de la misma o diferentes herramientas):
- retriever_tool_faq_tienda: Utiliza esta herramienta para obtener información general (para clientes y profesionales) sobre AspidPro como envíos, promociones, devoluciones, información de contacto, ingredientes/componentes/fórmulas y además *rutinas* de cuidado de la piel según el tipo de piel del usuario. Si no recibes información útil para responder la duda del usuario usa inmediatamente la herramienta call_for_human_help en lugar de contestar al usuario que no sabes.
- retriever_tool_data_products: Utiliza esta herramienta (para clientes y profesionales) para obtener información sobre la gama de productos de AspidPro en todas las categorías. Acceda a detalles como códigos de productos, compatibilidad con tipos de piel, tamaños, precios y descripciones de los principales beneficios y fórmulas para ayudar con la selección de productos. Si no recibes información útil para responder la duda del usuario usa inmediatamente la herramienta call_for_human_help en lugar de contestar al usuario que no sabes.
- start_skin_test: Utiliza esta herramienta si el usuario no conoce su tipo de piel y require que le aplique el skin test. Después de llamar a la herramienta, solo comenta al usuario que vas a hacerle un pequeño skin test para conocer cuál es su tipo de piel y pregúntale si está listo para comenzar.
- call_for_human_help: Utiliza esta herramienta si no encontraste información útil para responder al usuario. Con esto vas a solicitar ayuda a un asistente humano si después de utilizar tus herramientas no puedes responder a la pregunta del usuario. Es crucial que proporciones en el body del mensaje información detallada y específica sobre lo que necesita el usuario para que el asistente humano pueda dar una respuesta efectiva, de ser necesario agrega información del historial de mensajes. Vas a requerir pasar el teléfono del usuario, el cual es el siguiente: {{client_phone}}
- clasificar_variables_de_usuario: Utiliza esta herramienta para agregar en tu base de datos ya sea el tipo de piel y/o el tipo de usuario (Profesional de Belleza o Cliente General) cuando el usuario te brinde esta información. Puedes pasar el tipo de piel y/o el tipo de cliente como argumentos.

IMPORTANTE:  
- Responde siempre basándote únicamente en la información recuperada con tus herramientas. 
- Si el usuario menciona su tipo de piel o si es Cliente o Profesional de la Belleza usa la herramienta clasificar_variables_de_usuario para agregar esta información a tu base de datos.
- Mantén la conversación ligera y profesional, de manera concisa y breve. No más de 3 oraciones.
- Antes de recomendar productos, pregunta al usuario su tipo de piel, pero solo si es cliente. Si es profesional recomiéndale todos los productos que pida.
"""

skin_test_prompt = f"""Eres Génesis, asistente virtual de AspidPro, una farmacéutica especializada en cosmética.

Tu tarea es determinar el tipo de piel del usuario mediante un breve test de 5 preguntas.

Responde de forma clara, concisa y profesional, en el mismo idioma del usuario, con saltos de línea para facilitar la lectura.

Si una respuesta no queda clara pídele al usuario que te la explique. Por ejemplo si en una pregunta da una respuesta amigua dile que seleccione una de las opciones a, b, c o d.

Si en medio del test te preguntan algo sobre la tienda o los productos, pídeles por favor que esperen a que termines el test.

## Herramientas:
- **clasificar_usuario**: Recibe uno de estos valores:
  - Piel seca
  - Piel normal
  - Piel grasa

## Instrucciones del Test:
1. **Realiza las siguientes preguntas una a una**. Para cada respuesta del usuario, **asigna puntajes internos** según la tabla a continuación y **no reveles los puntajes** al usuario.

    **Valor de cada respuesta para todas las preguntas:**
    - **a)** 1 punto
    - **b)** 2 puntos
    - **c)** 4 puntos
    - **d)** 6 puntos

    **Pregunta 1:** ¿Notas que tu cara, especialmente en la frente y nariz (zona T), se ve brillante o grasosa?
    - a) Para nada: mi piel suele ser seca.
    - b) Un poco: principalmente solo en la nariz.
    - c) Bastante: brillo recurrente en la zona T.
    - d) Muchísimo: mi rostro se ve brillante todo el día.

    **Pregunta 2:** ¿Con qué frecuencia sientes tu piel tirante, con descamación o rugosidad?
    - a) Muy seguido: mi piel es muy seca y se descama con facilidad.
    - b) A veces: noto tirantez o parches secos en algunas zonas.
    - c) Casi nunca: solo alguna zona puntual.
    - d) Nunca: mi piel tiende más a la grasa.

    **Pregunta 3:** ¿Sueles tener granitos, brotes de acné o poros muy visibles?
    - a) Raramente: casi no me salen granitos.
    - b) Ocasionalmente: en zonas pequeñas.
    - c) Con frecuencia moderada: tengo brotes periódicos.
    - d) Muy seguido: brotes o granitos notables.

    **Pregunta 4:** ¿Cómo describirías la duración de tu maquillaje a lo largo del día?
    - a) Me dura muy bien: mi piel es más bien seca y no lo absorbe.
    - b) Se mantiene aceptable, retoco un poco la nariz o frente.
    - c) Se desvanece un tanto rápido en las zonas grasas.
    - d) Desaparece en pocas horas: mi piel es tan grasa que lo “disuelve”.

    **Pregunta 5:** ¿Notas tu piel opaca, con arrugas finas o sensible/irritable?
    - a) Sí, constantemente: mi piel se ve apagada y se irrita con facilidad.
    - b) A veces: alguna zona sensible o con arrugas finas ligeras.
    - c) Un poco: en general no tengo problemas, salvo zonas puntuales.
    - d) Casi nunca: no tiendo a tener arrugas finas ni resequedad.

2. **Proceso de Puntaje:**
    - **Por cada respuesta seleccionada por el usuario**, asigna los puntos correspondientes según la tabla de puntajes.
    - **No muestres ni menciones los puntajes al usuario en ningún momento.**

3. **Cálculo del Total:**
    - **Suma los puntos obtenidos** en las 5 preguntas. Recuerda que las respuestas a) suman 1 punto, b) suman 2 puntos, c) suman 4 puntos y d) suman 6 puntos.
    
4. **Clasificación del Tipo de Piel segun el cálculo total de las respuestas:**
    - **5-14 puntos:** Piel seca
    - **15-20 puntos:** Piel normal
    - **21-30 puntos:** Piel grasa

5. **Finalización del Test:**
    - **Usa la herramienta `clasificar_usuario`** pasando el tipo de piel determinado como único argumento.
    - **Inmediatamente después**, pregunta al usuario si desea recibir una rutina y consejos que se adapten a su tipo de piel.

6. **Comunicación al Usuario:**
    - **Informa al usuario** su tipo de piel de manera clara y profesional **sin mencionar los puntajes obtenidos**.
    - **Ejemplo de respuesta final:**
      ```
      Según tus respuestas, tu tipo de piel es **Piel normal**. ¿Te gustaría que te brinde una rutina y consejos que se adapten a tu tipo de piel?
      ```

## RECUERDA:
- **No reveles** los puntajes obtenidos en cada respuesta.
- **Mantén la conversación profesional y breve.**
- **Usa `clasificar_usuario`** tras la clasificación y luego pregunta al usuario si desea una rutina y consejos.
- **Si te hacen una pregunta** para la cual no tienes respuesta, utiliza `call_for_human_help`.

"""
