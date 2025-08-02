# src/rag/conversational_chain.py
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain.chat_models import ChatOpenAI          # or your provider

from .tools import retrieve

llm = ChatOpenAI(model_name="gpt-4o-mini")            # pick your model

def query_or_respond(state: MessagesState):
    """ask LLM to choose: answer or call tool."""
    msg = llm.bind_tools([retrieve]).invoke(state["messages"])
    return {"messages": [msg]}

def generate(state: MessagesState):
    """final answer after retrieval."""
    # collect ToolMessages just produced
    docs = "\n\n".join(m.content for m in state["messages"] if m.type == "tool")
    system = SystemMessage(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs}"
    )
    convo = [m for m in state["messages"]
             if m.type in ("human", "system") or (m.type == "ai" and not m.tool_calls)]
    response = llm.invoke([system] + convo)
    return {"messages": [response]}

# Assemble the graph
graph = StateGraph(MessagesState)
graph.add_node(query_or_respond)
graph.add_node(ToolNode([retrieve]), name="tools")
graph.add_node(generate)
graph.set_entry_point("query_or_respond")
graph.add_conditional_edges("query_or_respond", tools_condition,
                            {END: END, "tools": "tools"})
graph.add_edge("tools", "generate")
graph.add_edge("generate", END)
chat_rag = graph.compile()
