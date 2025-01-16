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
from langgraph.graph import MessagesState, START
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from datetime import datetime

from agents import react_prompt, llm, tools, skin_test_prompt, tools_for_skin_test


class State(MessagesState):
    summary: str
    message_type: str
    num_llamada: int
    tipo_de_cliente: str  # Paciente o Cliente
    tipo_de_piel: str  # Del Skin Test
    atencion_humana: bool
    aplicar_skin_test: bool
    client_phone: str


# EDGE
def requires_skin_test(state: State):
    # If aplicar_skin_test is True, go to "skin_test_node", otherwise to "call_model"
    if state.get("aplicar_skin_test", False):
        return "skin_test_node"
    else:
        return "call_model"


# NODE


def call_model(state: State):
    print("NODE call_model")

    # Existing summary and prompt logic
    summary = state.get("summary", "")
    num_llamada = state.get("num_llamada", 0)
    tipo_de_cliente = state.get("tipo_de_cliente", "")
    tipo_de_piel = state.get("tipo_de_piel", "")
    client_phone = state.get("client_phone", "")
    current_datetime = datetime.now().strftime(
        "Hoy es %A, %d de %B de %Y a las %I:%M %p."
    )

    prompt_with_time = react_prompt.format(
        tipo_de_cliente=tipo_de_cliente,
        tipo_de_piel=tipo_de_piel,
        current_datetime=current_datetime,
        client_phone=client_phone,
    )
    if summary:
        # Add summary to system message
        system_message_summary = f"Resumen de la conversación anterior: {summary}"
        # Append summary to any newer messages
        messages = [
            SystemMessage(content=prompt_with_time + system_message_summary)
        ] + state["messages"]
    else:
        messages = [SystemMessage(content=prompt_with_time)] + state["messages"]

    # Bind tools to LLM and invoke
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)

    # Check for tools
    # Initialize the keys we might update
    state["atencion_humana"] = state.get("atencion_humana", False)
    state["aplicar_skin_test"] = state.get("aplicar_skin_test", False)

    # Check if there's any tool call in the AI response
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            # If LLM calls for a human's help
            if tool_name == "call_for_human_help":
                print("NODE REASONER: Se habló a Humano")
                # This sets the atencion_humana flag
                state["atencion_humana"] = True

            # If LLM calls for a skin test
            elif tool_name == "start_skin_test":
                print("NODE REASONER: Se inicia skin test")
                # This sets the aplicar_skin_test flag
                state["aplicar_skin_test"] = True

            # If LLM calls to classify variables
            elif tool_name == "clasificar_variables_de_usuario":
                print("NODE REASONER: Clasificando variables del usuario")
                # Extract and update tipo_de_piel and tipo_de_cliente from tool arguments
                if "tipo_de_piel" in tool_args:
                    tipo_de_piel = tool_args["tipo_de_piel"]
                    state["tipo_de_piel"] = tipo_de_piel
                if "tipo_de_cliente" in tool_args:
                    tipo_de_cliente = tool_args["tipo_de_cliente"]
                    state["tipo_de_cliente"] = tipo_de_cliente

    # Return updated state and messages
    if num_llamada != 0:
        return {
            "messages": response,
            "message_type": "text",
            "tipo_de_piel": tipo_de_piel,
            "tipo_de_cliente": tipo_de_cliente,
            "atencion_humana": state["atencion_humana"],
            "aplicar_skin_test": state["aplicar_skin_test"],
        }
    else:
        return {
            "messages": response,
            "message_type": "image",
            "tipo_de_piel": tipo_de_piel,
            "tipo_de_cliente": tipo_de_cliente,
            "atencion_humana": state["atencion_humana"],
            "aplicar_skin_test": state["aplicar_skin_test"],
        }


def skin_test_node(state: State):
    print("NODE skin_test_node")

    # Existing summary and prompt logic
    summary = state.get("summary", "")
    num_llamada = state.get("num_llamada", 0)

    if summary:
        # Add summary to system message
        system_message_summary = f"Resumen de la conversación anterior: {summary}"
        # Append summary to any newer messages
        messages = [
            SystemMessage(content=skin_test_prompt + system_message_summary)
        ] + state["messages"]
    else:
        messages = [SystemMessage(content=skin_test_prompt)] + state["messages"]

    # Bind the tools (call_for_human_help, clasificar_usuario) to the LLM and invoke
    llm_with_tools = llm.bind_tools(tools_for_skin_test)
    response = llm_with_tools.invoke(messages)

    # Initialize the keys we might update
    state["atencion_humana"] = state.get("atencion_humana", False)
    state["tipo_de_piel"] = state.get("tipo_de_piel", "")
    state["aplicar_skin_test"] = state.get("aplicar_skin_test", True)

    # Check if there's any tool call in the AI response
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            # If LLM calls for a human's help
            if tool_name == "call_for_human_help":
                print("NODE REASONER: Se habló a Humano")
                # This sets the atencion_humana flag
                state["atencion_humana"] = True

            # If LLM attempts to classify the user's skin type
            elif tool_name == "clasificar_usuario":
                print("NODE REASONER: Se completó test de skin")
                # The LLM will pass something like {"tipo_de_piel": "Piel grasa"}
                tipo_de_piel = tool_args.get("tipo_de_piel", "")
                state["tipo_de_piel"] = tipo_de_piel
                state["aplicar_skin_test"] = False

    # Return updated state plus messages, using the num_llamada logic
    if num_llamada != 0:
        return {
            "messages": response,
            "message_type": "text",
            "tipo_de_piel": state["tipo_de_piel"],
            "atencion_humana": state["atencion_humana"],
            "aplicar_skin_test": state["aplicar_skin_test"],
        }
    else:
        return {
            "messages": response,
            "message_type": "image",
            "tipo_de_piel": state["tipo_de_piel"],
            "atencion_humana": state["atencion_humana"],
            "aplicar_skin_test": state["aplicar_skin_test"],
        }


def summarize_conversation(state: State):
    print("NODE summarize_conversation")
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Begin filtering logic
    all_messages = state["messages"]
    last_human_index = None
    for i in range(len(all_messages) - 1, -1, -1):
        if isinstance(all_messages[i], HumanMessage):
            last_human_index = i
            break

    if last_human_index is None:
        # No human message found
        delete_messages = []
        return {"summary": response.content, "messages": delete_messages}

    end_index = min(last_human_index + 4, len(all_messages))
    candidates = all_messages[last_human_index:end_index]

    # Check for AIMessage with tool_calls
    last_ai_tool_index = None
    for idx, msg in enumerate(candidates):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            last_ai_tool_index = idx

    if last_ai_tool_index is not None:
        global_ai_index = last_human_index + last_ai_tool_index
        if global_ai_index + 1 < len(all_messages) and isinstance(
            all_messages[global_ai_index + 1], ToolMessage
        ):
            tool_msg = all_messages[global_ai_index + 1]
            if tool_msg not in candidates:
                insertion_pos = last_ai_tool_index + 1
                candidates = (
                    candidates[:insertion_pos] + [tool_msg] + candidates[insertion_pos:]
                )
        else:
            # Remove the AIMessage with tool_calls if it doesn't have a matching ToolMessage
            candidates = [
                m for m in candidates if m is not candidates[last_ai_tool_index]
            ]

    # Ensure the final conversation starts with a HumanMessage
    first_human_pos = None
    for idx, msg in enumerate(candidates):
        if isinstance(msg, HumanMessage):
            first_human_pos = idx
            break

    if first_human_pos is None:
        final_messages = candidates
    else:
        final_messages = candidates[first_human_pos:]

    delete_messages = []
    for m in state["messages"]:
        if m not in final_messages:
            print(
                f"Deleting message with ID: {m.id}, Content: {m.content}, Kwargs: {m.additional_kwargs}"
            )
            delete_messages.append(RemoveMessage(id=m.id))

    return {"summary": response.content, "messages": delete_messages}


#
# REPLACE dummy_node WITH clear_tool_messages
#
def clear_tool_messages(state: State):
    print("NODE clear_tool_messages")
    """
    This node deletes:
       1) Any AIMessage that contains a `tool_calls` attribute
       2) Any ToolMessage in the conversation

    The idea is to remove the entire "tool call + tool response"
    so we don't leave the LLM in a state where an AI tool-call
    references a missing tool message.
    """

    # We collect messages to delete
    messages_to_remove = []
    num_llamada = state.get("num_llamada", 0)
    # If an AI message has .tool_calls, it must be followed by a ToolMessage.
    # We'll remove them both so there's no mismatch.
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            print(
                f"Deleting AIMessage (tool call) with ID: {msg.id}, "
                f"Content: {msg.content}, Kwargs: {msg.additional_kwargs}"
            )
            messages_to_remove.append(RemoveMessage(id=msg.id))

        elif isinstance(msg, ToolMessage):
            print(
                f"Deleting ToolMessage with ID: {msg.id}, "
                f"Content: {msg.content}, Kwargs: {msg.additional_kwargs}"
            )
            messages_to_remove.append(RemoveMessage(id=msg.id))

    num_llamada += 1
    # Return only the messages to remove, no need to update the summary
    return {"messages": messages_to_remove, "num_llamada": num_llamada}


def should_continue(state: State):
    print("EDGE should_continue")
    messages = state["messages"]

    # If there are more than 40 messages, then we summarize the conversation
    if len(messages) > 40:
        return "summarize_conversation"

    return END


# Setup workflow
workflow = StateGraph(State)

workflow.add_node("call_model", call_model)
workflow.add_node("skin_test_node", skin_test_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("tools_for_skin_test", ToolNode(tools_for_skin_test))
tools_for_skin_test
workflow.add_node("clear_tool_messages", clear_tool_messages)
workflow.add_node("summarize_conversation", summarize_conversation)

workflow.add_conditional_edges(
    START,
    requires_skin_test,
    {"skin_test_node": "skin_test_node", "call_model": "call_model"},
)
# workflow.add_edge("skin_test_node", END)
workflow.add_conditional_edges(
    "skin_test_node", tools_condition, {"tools": "tools_for_skin_test", END: END}
)
workflow.add_edge("tools_for_skin_test", "skin_test_node")
# workflow.set_entry_point("call_model")
workflow.add_conditional_edges(
    "call_model", tools_condition, {"tools": "tools", END: "clear_tool_messages"}
)
workflow.add_edge("tools", "call_model")


workflow.add_conditional_edges(
    "clear_tool_messages",
    should_continue,
    {"summarize_conversation": "summarize_conversation", END: END},
)
workflow.add_edge("summarize_conversation", END)


# MEMORY

os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/graphs/your_database_file.db", check_same_thread=False)
memory = SqliteSaver(conn)

react_graph = workflow.compile(checkpointer=memory)


def call_model(messages, client_phone, config):
    # Stream events from the graph
    events = react_graph.stream(
        {"messages": messages, "client_phone": client_phone},
        config,
        stream_mode="values",
    )

    response_content = None
    message_type = None

    for event in events:
        if "messages" in event and event["messages"]:
            last_message = event["messages"][-1]
            response_content = last_message.content
            # Access the message_type from the event
            message_type = event.get("message_type", None)

    return response_content, message_type
