# airregio_graph_crm.py
import operator
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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


os.environ["AIRTABLE_API_KEY"] = os.getenv("AIRTABLE_API_KEY")

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
    "Search and return information about Aspid Pro shipping, returns, contact information, ingredients/components/formulas, and skincare routine.",
)

retriever_tool_data_products = create_retriever_tool(
    retriever_info_productos,
    "retriever_info_productos",
    "Retrieve comprehensive information about Aspid Pro's product range across all categories. Access details such as product codes, skin type compatibility, sizes, prices, and descriptions of key benefits and formulations to assist with product selection and skincare routines.",
)

tools = [retriever_tool_faq_tienda, retriever_tool_data_products]

react_prompt = f"""Eres Assy, un asistente virtual de Aspid Pro. 

Eres un asistente profesional y amable que trabaja para Aspid Pro, una farmacéutica especializada en cosmética. 

Tu principal tarea es ofrecer al usuario información sobre los productos de Aspid Pro.

Responde de manera concisa. No más de 3 oraciones.

Responde en el mismo idioma en el que el usuario se comunique contigo.  

Asegúrate de mantener la conversación amistosa y clara, añadiendo saltos de línea para que los mensajes sean fáciles de leer. Pero mantén tu respuesta concisa.

Always answer based only on the information retrieved with your tools.

Si no sabes la respuesta di que no tienes información al respecto pero que un asistente humano se comunicará en breve con el usuario para ayudarlo.

Interpreta cualquier información ambigua sobre la fecha y la hora, considerando el siguiente contexto temporal:
{{current_datetime}}

## Herramientas disponibles:
- retriever_tool_faq_tienda: Utiliza esta herramienta para obtener información general sobre envíos, devoluciones, información de contacto, ingredientes/componentes/fórmulas y rutinas de cuidado de la piel de Aspid Pro. 
- retriever_tool_data_products: Utiliza esta herramienta para obtener información sobre la gama de productos de Aspid Pro en todas las categorías. Acceda a detalles como códigos de productos, compatibilidad con tipos de piel, tamaños, precios y descripciones de los principales beneficios y fórmulas para ayudar con la selección de productos y las rutinas de cuidado de la piel.

RECUERDA:  
- Mantén la conversación ligera y profesional, de manera concisa y breve. No más de 3 oraciones.
- Responde siempre basándote únicamente en la información recuperada con tus herramientas. 
"""
