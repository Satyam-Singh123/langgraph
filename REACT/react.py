import os
import json
import re
import requests
import ast
import traceback

#typing PyDantic
from typing import Any, Optional, Dict, Callable

#AgentState
from typing import Annotated, List, TypedDict, Optional
import operator

#get .env 
from dotenv import load_dotenv
load_dotenv()

# Watsonx imports
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# LangGraph + messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage


REACT_PROMPT = """ You are a specialized agent. Your goal is to answer the user's request. 
You have access to the following tool with name: {tool_name} and description: {tool_description}.

You must respond in one of two formats:

1. Final Answer
Think: I have Enough Information to Answer the use Query
Action: Final Answer
Action Input: Final Answer from context goes here

2. Tool Call
Think: I need to use tool to get Information
Action: {tool_name}
Action Input: {{"query" : "the search term goes here" }}

Begin.
"""

def get_llm(model_id: str = "meta-llama/llama-3-3-70b-instruct",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> ModelInference:
    
    creds = Credentials(api_key=os.getenv("WATSONX_API_KEY"),  url=os.getenv("WATSONX_URL"))
    client = APIClient (credentials=creds)

    return ModelInference(
        model_id=model_id,
        params={GenParams.MAX_NEW_TOKENS:max_new_tokens, GenParams.TEMPERATURE:temperature},
         project_id=os.getenv("WATSONX_PROJECT_ID"), 
         api_client=client
        

    )


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    final_output:Optional[str]
    action_name:Optional[str]
    action_input:Optional[str]


def react_llm_node(
        state:AgentState,
        tool_name: str,
        tool_description: str,
        model_id: str = "meta-llama/llama-3-3-70b-instruct",       
):
    
    conversation=""
    for msg in state["messages"]:
        if isinstance(msg,AIMessage):
            conversation += f"Assistant: {msg.content}\n"
            
        elif isinstance(msg, HumanMessage):
            conversation += f"User: {msg.content}\n"

        elif isinstance(msg, ToolMessage):
            conversation += f"Tool: {msg.content}\n"

    prompt = REACT_PROMPT.format(
        tool_name=tool_name,
        tool_description=tool_description
    ) + "\n" + conversation

    llm = get_llm(model_id=model_id)

    llm_result= llm.generate_text(prompt)
    if isinstance(llm_result, dict) and "results" in llm_result:
        react_llm_result = llm_result["results"][0]["generated_text"]
    elif isinstance(llm_result, str):
        react_llm_result = llm_result
    else:
        react_llm_result = str(llm_result)


    return {"messages": [AIMessage(content=react_llm_result)]}
    pass
    


def react_llm_parser_and_decide_node(state:AgentState):
   
    last = state["messages"][-1]
    text = last.content or ""
    m = re.search(
        r"Action:\s*(.+?)\s*\nAction Input:\s*(.+)",
        text,
        re.DOTALL | re.MULTILINE
    )

    if not m:
        return {"action_name": None, "action_input": None}

    action_name = m.group(1).strip()
    action_input_raw = m.group(2).strip()

    try:
        action_input = json.loads(action_input_raw)
    except:
        try:
            action_input = ast.literal_eval(action_input_raw)
        except:
            action_input = {"raw": action_input_raw}
            
    if action_name.lower() == "final answer":
        return {"action_name": None, "action_input": action_input}
    
    return {"action_name":action_name, "action_input": action_input}

def tavily_search(query: str, top_k: int = 5) -> dict:
    """
    Real Tavily search call.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    endpoint = os.getenv("TAVILY_ENDPOINT")

    if not api_key or not endpoint:
        raise RuntimeError("Missing Tavily environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {"query": query, "top_k": top_k}
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()

    return resp.json()

TOOLS: Dict[str, Callable[..., Any]] = {
    "tavily_search": tavily_search,
}

def tool_executor_node(state:AgentState):
    name = state.get("action_name")
    data = state.get("action_input") or {}

    if not name:
        return {}

    tool_fn = TOOLS.get(name)
    if not tool_fn:
        result_text = f"Unknown tool: {name}"
    else:
        try:
            if isinstance(data, dict) and "query" in data:
                result = tool_fn(query=data["query"], **{k: v for k, v in data.items() if k != "query"})
            else:
                result = tool_fn(**data)
            result_text = json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            result_text = json.dumps({"error": str(e), "trace": traceback.format_exc()})

    # FIX: ToolMessage requires tool_call_id
    tool_msg = ToolMessage(
        content=result_text,
        tool_call_id=name
    )

    return {"messages": [tool_msg], "action_name": None, "action_input": None}

def extract_final_answer(state: AgentState) -> Optional[str]:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and "Final Answer" in msg.content:
            m = re.search(r"Action Input:\s*(.+)$", msg.content, re.S)
            if m:
                return m.group(1).strip()
            return {"final_output": msg.content}
    return None

# -----------------------------------------------------------
# Build LangGraph
# -----------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node(
        "react_llm",
        lambda s: react_llm_node(
            s,
            tool_name="tavily_search",
            tool_description="Searches the internet and returns relevant info via Tavily."
        ),
    )

    graph.add_node("router", react_llm_parser_and_decide_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("extract_final_answer", extract_final_answer)

    graph.add_edge(START, "react_llm")
    graph.add_edge("react_llm", "router")


    def route_fn(state: AgentState):
        return "extract_final_answer" if state.get("action_name") is None else "tool"

    graph.add_conditional_edges(
        "router",
        route_fn,
        {"extract_final_answer": "extract_final_answer", "tool": "tool_executor"}
      
    )

    graph.add_edge("extract_final_answer", END)



    graph.add_edge("tool_executor", "react_llm")
    return graph

def generate_transcript(topic: str, llm: Optional[ModelInference] = None) -> str:
    if llm is None:
        llm = get_llm()

    prompt = f"""
Generate a clear, spoken-style transcript for a YouTube trending video for kids about:
"{topic}"

Requirements:
- 400–700 words
- Conversational tone
- Clear structure: hook → points → ending
- No directions or metadata, only spoken words
"""

    response = llm.generate_text(prompt)

    if isinstance(response, dict) and "results" in response:
        return response["results"][0].get("generated_text", "")
    elif isinstance(response, str):
        return response
    return str(response)

def main():
    graph = build_graph()
    app = graph.compile()

    user_query = input("Enter your query/topic: ").strip()

    state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "action_name": None,
        "action_input": None,
        "raw_output": None,
    }

    result = app.invoke(state)
    print("result", result)
    final =result

    print("Final: ", final)
    if final:
        final=final.get("final_output")
        print("\n==== Final Answer ====\n")
        print(final)

        print("\n==== Generating Transcript ====\n")
        # topic = user_query
        transcript = generate_transcript(final)
        print("Transcript : ", transcript)

    else:
        print("\n==== No Final Answer Found ====\n")
        for m in result["messages"]:
            print(type(m).__name__, ":", m.content)



