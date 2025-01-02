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
from langgraph.graph import MessagesState
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


from datetime import datetime

from agents import react_prompt, llm, tools


class State(MessagesState):
    summary: str


def call_model(state: State):
    print("NODE call_model")

    # Initialize return values from the state

    # Ensure we only process ToolMessages
    # last_message = state["messages"][-1]
    # print(f"call_node LAST MESAGE {last_message}")

    # Existing summary and prompt logic
    summary = state.get("summary", "")

    # If there is a summary, include it in the system message
    current_datetime = datetime.now().strftime(
        "Hoy es %A, %d de %B de %Y a las %I:%M %p."
    )

    prompt_with_time = react_prompt.format(current_datetime=current_datetime)
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

    # Return the updated state values along with the LLM response
    return {
        "messages": response,
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
workflow.add_node("dummy_node", dummy_node)
workflow.add_node("summarize_conversation", summarize_conversation)


workflow.set_entry_point("call_model")
workflow.add_conditional_edges(
    "call_model", tools_condition, {"tools": "tools", END: "dummy_node"}
)
workflow.add_edge("tools", "call_model")


workflow.add_conditional_edges(
    "dummy_node",
    should_continue,
    {"summarize_conversation": "summarize_conversation", END: END},
)
workflow.add_edge("summarize_conversation", END)


# MEMORY

# Ensure the 'data' directory exists
os.makedirs("data", exist_ok=True)

# Create an SQLite connection with check_same_thread=False
conn = sqlite3.connect("data/graphs/your_database_file.db", check_same_thread=False)

memory = SqliteSaver(conn)

react_graph = workflow.compile(checkpointer=memory)


def call_model(messages, config):
    # Do not include "messages" in the initial state
    events = react_graph.stream(
        {
            "messages": messages,
        },
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
