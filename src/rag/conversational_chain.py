from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

from rag.tools import retrieve
import os
from langchain_core.messages import ToolMessage
USE_TOOLS = os.getenv("RAG_USE_TOOLS", "1") == "1"
# If RAG_USE_TOOLS is set to 0, we will not use the tool
# Local LLM served by Ollama free
# Change the model tag if you want to pull something else
llm = ChatOllama(model="mistral:7b", temperature=0.2)


# let the LLM decide to answer or call the tool
def query_or_respond(state: MessagesState):
    if USE_TOOLS:
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    user_message = state["messages"][-1]
    tool_text, _ = retrieve.invoke(user_message.content)
    return {"messages": [
        ToolMessage(
            content=tool_text,
            name="retrieve",
            tool_call_id="manual")]
    }

# if a tool was requested, execute it 
tools = ToolNode([retrieve])


# Step 3 — craft the final answer using any retrieved context 
def generate(state: MessagesState):
    tool_blocks = [message.content for message in state["messages"] if message.type == "tool"]

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        + "\n\n".join(tool_blocks)
    )

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = ([SystemMessage(system_message_content)] + conversation_messages)
    response = llm.invoke(prompt)
    return {"messages": [response]}


# Build the LangGraph exactly like the tutorial
graph = StateGraph(MessagesState)
graph.add_node(query_or_respond)    # entry node
graph.add_node(tools, name="tools") # retrieval node
graph.add_node(generate)    # generation node

graph.set_entry_point("query_or_respond")
graph.add_conditional_edges(
    "query_or_respond", tools_condition, {END: END, "tools": "tools"}
)
graph.add_edge("tools", "generate")
graph.add_edge("generate", END)

chat_rag = graph.compile()