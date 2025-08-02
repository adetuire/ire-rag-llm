from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

from rag.tools import retrieve

# Local LLM served by Ollama free
llm = ChatOllama(model="mistral:7b", temperature=0.2)


# let the LLM decide to answer or call the tool
def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# if a tool was requested, execute it 
tools = ToolNode([retrieve])


# Step 3 — craft the final answer using any retrieved context 
def generate(state: MessagesState):
    context_blocks = [message.content for message in state["messages"] if message.type == "tool"]
    if not context_blocks:
        return {"messages": ["No relevant context found."]}

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        + "\n\n".join(context_blocks)
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
graph.add_node(query_or_respond)
graph.add_node(tools, name="tools")
graph.add_node(generate)

graph.set_entry_point("generate_ai_message")
graph.add_conditional_edges(
    "generate_ai_message", tools_condition, {END: END, "tools": "tools"}
)
graph.add_edge("tools", "generate")
graph.add_edge("generate", END)

chat_rag = graph.compile()