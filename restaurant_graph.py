# restaurant_graph.py
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


from datetime import datetime

from agents_restaurant import (
    react_prompt,
    llm,
    tools,
    info_extraction_prompt,
    extract_tools,
    recordar_informacion_importante,
)


class State(MessagesState):
    extracted_messages: list
    summary: str
    id: str  # From AirTable <---------- Retrieve from API
    booked_status: bool
    name: str
    phone: str
    email: str
    persons_number: int
    date: str
    time: str
    requests: str


import json

import json
from typing import Optional


def call_model(state: State):
    print("NODE call_model")

    # Initialize return values from the state
    id = state.get("id", "")
    booked_status = state.get("booked_status", False)
    name = state.get("name", "")
    phone = state.get("phone", "")
    email = state.get("email", "")
    persons_number = state.get("persons_number", None)
    date = state.get("date", "")
    time = state.get("time", "")
    requests = state.get("requests", "")

    # Ensure we only process ToolMessages
    last_message = state["messages"][-1]
    print(f"call_node LAST MESAGE {last_message}")
    if isinstance(last_message, ToolMessage):
        print("call_node: It is a Tool Message")
        # Parse ToolMessage to check tool name and response
        try:
            tool_response = json.loads(last_message.content)
            print(f"call_node: Tool_response: {tool_response}")
            tool_name = last_message.name

            if tool_name == "add_user_to_restaurant_db" and tool_response.get(
                "success", False
            ):
                # Extract the ID from the response
                record_id = tool_response["record"]["id"]

                # Update id and booked_status
                id = record_id
                booked_status = True
                print(f"Reservation successful. ID: {record_id}")

        except json.JSONDecodeError:
            print("Error decoding tool message content. Skipping tool processing.")
        except KeyError as e:
            print(f"Missing key in tool response: {e}. Skipping tool processing.")

    # Existing summary and prompt logic
    summary = state.get("summary", "")

    current_datetime = datetime.now().strftime(
        "Hoy es %A, %d de %B de %Y a las %I:%M %p."
    )
    content_prompt_with_time = react_prompt.format(
        name=name,
        phone=phone,
        email=email,
        persons_number=persons_number,
        date=date,
        time=time,
        requests=requests,
        current_datetime=current_datetime,
        id=id,
        booked_status=booked_status,
    )

    # If there is a summary, include it in the system message
    if summary:
        # Add summary to system message
        system_message_summary = f"Resumen de la conversación anterior: {summary}"
        # Append summary to any newer messages
        messages = [
            SystemMessage(content=content_prompt_with_time + system_message_summary)
        ] + state["messages"]
    else:
        messages = [SystemMessage(content=content_prompt_with_time)] + state["messages"]

    # Bind tools to LLM and invoke
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)

    # Return the updated state values along with the LLM response
    return {
        "messages": response,
        "id": id,
        "booked_status": booked_status,
    }


def extract_data(state: State):
    print("NODE extract_data")

    # Retrieve existing known attributes from state
    name = state.get("name", "")
    phone = state.get("phone", "")
    email = state.get("email", "")
    persons_number = state.get("persons_number", None)
    date = state.get("date", "")
    time = state.get("time", "")
    requests = state.get("requests", "")

    last_message = state["messages"][-1]

    # Find the last HumanMessage in the state
    def find_last_human_message(state: MessagesState):
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                return message
        return None

    last_human_message = find_last_human_message(state)

    # If a HumanMessage is found, slice the messages list up to (and including) it
    if last_human_message:
        last_human_index = state["messages"].index(last_human_message)
        filtered_messages = state["messages"][: last_human_index + 1]
    else:
        # If no HumanMessage is found, include all messages
        filtered_messages = state["messages"]

    # Create a new messages list up to the last HumanMessage
    current_datetime = datetime.now().strftime(
        "Hoy es %A, %d de %B de %Y a las %I:%M %p."
    )
    prompt = info_extraction_prompt.format(
        name=name,
        phone=phone,
        email=email,
        persons_number=persons_number,
        date=date,
        time=time,
        requests=requests,
        current_datetime=current_datetime,
    )
    messages = [SystemMessage(content=prompt)] + filtered_messages
    llm_with_tools = llm.bind_tools(extract_tools)
    ai_tool_message = llm_with_tools.invoke(messages)

    # Check if the AIMessage requests a tool call
    if hasattr(ai_tool_message, "tool_calls") and ai_tool_message.tool_calls:
        # Extract the tool call info
        tool_call = ai_tool_message.tool_calls[0]  # Assuming only one call
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})

        if tool_name == "recordar_informacion_importante":
            print("AI is requesting tool call: recordar_informacion_importante")

            # Extract arguments
            nombre_del_cliente = tool_args.get("nombre_del_cliente")
            telefono = tool_args.get("telefono")
            correo_electronico = tool_args.get("correo_electronico")
            numero_de_personas = tool_args.get("numero_de_personas")
            fecha = tool_args.get("fecha")
            hora = tool_args.get("hora")
            solicitudes_extra = tool_args.get("solicitudes_extra")

            # Call the tool
            tool_response = recordar_informacion_importante(
                nombre_del_cliente=nombre_del_cliente,
                telefono=telefono,
                correo_electronico=correo_electronico,
                numero_de_personas=numero_de_personas,
                fecha=fecha,
                hora=hora,
                solicitudes_extra=solicitudes_extra,
            )

            # Update state from the tool response
            name = tool_response.get("name", name)
            phone = tool_response.get("phone", phone)
            email = tool_response.get("email", email)
            persons_number = tool_response.get("persons_number", persons_number)
            date = tool_response.get("date", date)
            time = tool_response.get("time", time)
            requests = tool_response.get("requests", requests)

            print("Processed reservation data updated in state after tool call.")

    # Return the updated state without extracted_messages
    return {
        "name": name,
        "phone": phone,
        "email": email,
        "persons_number": persons_number,
        "date": date,
        "time": time,
        "requests": requests,
    }


def summarize_conversation(state: State):
    print("NODE summarize_conversation")
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history and invoke the LLM
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Begin filtering logic

    # 1. Find the last HumanMessage in state["messages"]
    all_messages = state["messages"]
    last_human_index = None
    for i in range(len(all_messages) - 1, -1, -1):
        if isinstance(all_messages[i], HumanMessage):
            last_human_index = i
            break

    if last_human_index is None:
        # No human message found, just keep everything
        # and return as the original code would, but no deletions
        delete_messages = []
        return {"summary": response.content, "messages": delete_messages}

    # 2. From that HumanMessage, keep up to 4 messages following it (including the HumanMessage itself)
    # But we might need more if there's a tool call AIMessage and a following ToolMessage.
    end_index = min(last_human_index + 4, len(all_messages))
    candidates = all_messages[last_human_index:end_index]

    # 3. Check for AIMessage with tool_calls in candidates.
    # If found, ensure the immediate next message is a ToolMessage and include it if missing.
    last_ai_tool_index = None
    for idx, msg in enumerate(candidates):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            last_ai_tool_index = idx

    if last_ai_tool_index is not None:
        # Ensure that the message after the AIMessage with tool_calls is a ToolMessage
        global_ai_index = last_human_index + last_ai_tool_index
        if global_ai_index + 1 < len(all_messages) and isinstance(
            all_messages[global_ai_index + 1], ToolMessage
        ):
            tool_msg = all_messages[global_ai_index + 1]
            if tool_msg not in candidates:
                # Insert the tool message right after the AIMessage with tool_calls
                insertion_pos = last_ai_tool_index + 1
                candidates = (
                    candidates[:insertion_pos] + [tool_msg] + candidates[insertion_pos:]
                )
        else:
            # If there's no ToolMessage after an AIMessage with tool_calls,
            # remove the AIMessage with tool_calls to avoid invalid sequence.
            candidates = [
                m for m in candidates if m is not candidates[last_ai_tool_index]
            ]

    # 4. Ensure the final conversation starts with a HumanMessage.
    first_human_pos = None
    for idx, msg in enumerate(candidates):
        if isinstance(msg, HumanMessage):
            first_human_pos = idx
            break

    if first_human_pos is None:
        # No HumanMessage found (unexpected because we started from one),
        # just return candidates as is, no deletions.
        final_messages = candidates
    else:
        final_messages = candidates[first_human_pos:]

    # 5. Determine which messages to delete
    # We only keep final_messages and delete the rest
    delete_messages = []
    for m in state["messages"]:
        if m not in final_messages:
            print(
                f"Deleting message with ID: {m.id}, Content: {m.content}, Kwargs: {m.additional_kwargs}"
            )
            delete_messages.append(RemoveMessage(id=m.id))

    # Return the summary and the messages to delete, as the original code does.
    return {"summary": response.content, "messages": delete_messages}


def dummy_node(state: State):
    print("NODE dummy_node")
    pass


def should_continue(state: State):
    """Return the next node to execute."""
    print("EDGE should_continue")
    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 18:
        return "summarize_conversation"

    # Otherwise we can just end
    return END


# Setup workflow
workflow = StateGraph(State)

workflow.add_node("call_model", call_model)
workflow.add_node("tools", ToolNode(tools))
# workflow.add_node("extract_tools", ToolNode(extract_tools))
# workflow.add_node("dummy_node", dummy_node)
workflow.add_node("extract_data", extract_data)
workflow.add_node("summarize_conversation", summarize_conversation)


workflow.set_entry_point("call_model")
workflow.add_conditional_edges(
    "call_model", tools_condition, {"tools": "tools", END: "extract_data"}
)
workflow.add_edge("tools", "call_model")

# workflow.add_edge("extract_data", "dummy_node")
workflow.add_conditional_edges(
    "extract_data",
    should_continue,
    {"summarize_conversation": "summarize_conversation", END: END},
)
workflow.add_edge("summarize_conversation", END)


# MEMORY

# Ensure the 'data' directory exists
os.makedirs("data/graphs_restaurant", exist_ok=True)

# Create an SQLite connection with check_same_thread=False
conn = sqlite3.connect(
    "data/graphs_restaurant/your_database_file.db", check_same_thread=False
)

memory = SqliteSaver(conn)

react_graph = workflow.compile(checkpointer=memory)


def call_model_restaurant_bot(messages, phone, config):
    # Do not include "messages" in the initial state
    events = react_graph.stream(
        {"messages": messages, "phone": phone},
        config,
        stream_mode="values",
    )

    response = None  # Initialize response
    for event in events:
        # Each event is a dictionary containing different stages of the graph execution
        if "messages" in event and event["messages"]:
            response = event["messages"][
                -1
            ].content  # Get the content of the last message

    return response  # Return the final response content


def call_model_from_messenger(messages, config):
    # Do not include "messages" in the initial state
    events = react_graph.stream(
        {"messages": messages},
        config,
        stream_mode="values",
    )

    response = None  # Initialize response
    for event in events:
        # Each event is a dictionary containing different stages of the graph execution
        if "messages" in event and event["messages"]:
            response = event["messages"][
                -1
            ].content  # Get the content of the last message

    return response  # Return the final response content
