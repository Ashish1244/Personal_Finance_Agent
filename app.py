import streamlit as st
import os
from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ==========================================
# 1. DEFINE LANGGRAPH STATE & SCHEMA
# ==========================================

class FinancialProfile(BaseModel):
    income: Optional[float] = Field(None, description="Monthly income or allowance in Rupees")
    expenses: Optional[float] = Field(None, description="Average monthly expenses in Rupees")
    reason_for_investment: Optional[str] = Field(None, description="The core goal or reason for wanting to invest")
    missing_question: Optional[str] = Field(None, description="A single polite question if any of the above 3 fields are missing. If none are missing, leave empty.")

class AgentState(TypedDict):
    messages: List[BaseMessage]
    profile: FinancialProfile
    initial_recommendation: Optional[str]
    current_step: str  # Tracks: "GATHERING", "VERIFICATION", "COMPLETED"

# Initialize Groq LLM using the active production model
UPGRADED_MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=UPGRADED_MODEL, temperature=0.2)

# ==========================================
# 2. DEFINE THE OPERATIONAL GRAPH NODES
# ==========================================

def gather_info_node(state: AgentState):
    """Steps 1, 2 & 3: Extracts profile variables and checks for missing entries."""
    messages = state["messages"]
    
    system_prompt = (
        "You are an information extraction assistant. Analyze the chat history between the user and the agent. "
        "Extract: income, expenses, and reason_for_investment.\n"
        "If any of these three fields are missing, generate a polite question to ask for ONE missing piece of data in the 'missing_question' field.\n"
        "If all three pieces of data are present, set 'missing_question' to empty or null."
    )
    
    structured_llm = llm.with_structured_output(FinancialProfile)
    updated_profile = structured_llm.invoke([AIMessage(content=system_prompt)] + messages)
    
    next_step = "VERIFICATION" if not updated_profile.missing_question else "GATHERING"
    return {"profile": updated_profile, "current_step": next_step}


def recommend_node(state: AgentState):
    """Step 4: Generates a brief, highly practical high-level allocation recommendation."""
    profile = state["profile"]
    
    # Calculate available savings capacity
    surplus = max(0.0, profile.income - profile.expenses)
    
    prompt = f"""
    You are an expert personal finance advisor for college students and young adults in India.
    Based on the profile below, calculate their monthly investable surplus (Income: ₹{profile.income} - Expenses: ₹{profile.expenses} = ₹{surplus}).
    
    Provide a highly practical, 2-3 sentence high-level recommendation of exactly how they should split this ₹{surplus} surplus using relatable Indian vehicles:
    - Emergency Cash (High-yield Savings Account / Liquid Fund)
    - Monthly Systematic Investment Plans (SIPs) in Equity/Index Mutual Funds for growth
    - Safe avenues like PPF or Fixed Deposits if their goal is short-term or low-risk.
    
    Match the breakdown perfectly to their investment reason: "{profile.reason_for_investment}".
    End the response by asking clearly: "Does this breakdown look good to you?"
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"initial_recommendation": response.content}

# --- ROUTING LOGIC ---
def route_after_gathering(state: AgentState):
    if state["current_step"] == "VERIFICATION":
        return "generate_recommendation"
    return "ask_more_questions"

# ==========================================
# 3. COMPILATION PIPELINE FUNCTION
# ==========================================

def build_langgraph_agent():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("gather_info", gather_info_node)
    workflow.add_node("recommend", recommend_node)
    
    workflow.set_entry_point("gather_info")
    
    workflow.add_conditional_edges(
        "gather_info",
        route_after_gathering,
        {
            "ask_more_questions": END,      
            "generate_recommendation": "recommend"
        }
    )
    workflow.add_edge("recommend", END)
    return workflow.compile()

# ==========================================
# 4. STREAMLIT INTERACTIVE UI LAYOUT
# ==========================================

st.set_page_config(page_title="Personal Finance Agent", page_icon="🤖")
st.title("🤖 Personal Finance Decision Agent")

onboarding_guidance = (
    "Hello! Let's build your practical, student-friendly investment blueprint.\n\n"
    "To get started, please tell me:\n"
    "1. **Your monthly income/allowance** (e.g., 'My income is ₹15,000')\n"
    "2. **Your monthly expenses** (e.g., 'I spend ₹8,000')\n"
    "3. **What you are saving or investing for** (e.g., 'Buying a laptop next year' or 'Starting an SIP for long-term growth')\n\n"
    "What are your current income and expense numbers?"
)

# Initialize data objects in persistent session slots
if "agent_app" not in st.session_state:
    st.session_state.agent_app = build_langgraph_agent()
    st.session_state.graph_state = {
        "messages": [AIMessage(content=onboarding_guidance)],
        "profile": FinancialProfile(),
        "initial_recommendation": "",
        "current_step": "GATHERING"
    }

# Loop and render conversation logs from persistent history storage
for msg in st.session_state.graph_state["messages"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle ongoing human interactions
if st.session_state.graph_state["current_step"] != "COMPLETED":
    
    if user_input := st.chat_input("Provide your details or feedback..."):
        # Display and record the text immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.graph_state["messages"].append(HumanMessage(content=user_input))
        
        with st.chat_message("assistant"):
            
            # --- PHASE 1: GATHERING & INITIAL SUGGESTION ---
            if st.session_state.graph_state["current_step"] == "GATHERING":
                with st.spinner("Processing profile data..."):
                    output = st.session_state.agent_app.invoke(st.session_state.graph_state)
                    st.session_state.graph_state.update(output)
                
                if st.session_state.graph_state["profile"].missing_question:
                    bot_msg = st.session_state.graph_state["profile"].missing_question
                    st.markdown(bot_msg)
                    st.session_state.graph_state["messages"].append(AIMessage(content=bot_msg))
                else:
                    bot_msg = st.session_state.graph_state["initial_recommendation"]
                    st.markdown(bot_msg)
                    st.session_state.graph_state["messages"].append(AIMessage(content=bot_msg))
            
            # --- PHASE 2: HUMAN INTERRUPT VERIFICATION GATEWAY ---
            elif st.session_state.graph_state["current_step"] == "VERIFICATION":
                with st.spinner("Analyzing your confirmation..."):
                    check_prompt = f"Analyze if this user response means they accept or agree with the plan: '{user_input}'. Respond strictly with only the word 'YES' or 'NO'."
                    verification = llm.invoke([HumanMessage(content=check_prompt)]).content.strip().upper()
                
                if "YES" in verification:
                    # Step 6: Create the detailed, highly relatable step-by-step roadmap
                    with st.spinner("Generating step-by-step implementation plan..."):
                        blueprint_prompt = f"""
                        The user approved your allocation strategy! Now provide a highly practical, relatable, step-by-step execution roadmap for a beginner in India.
                        
                        Based on their strategy ({st.session_state.graph_state['initial_recommendation']}), explain exactly:
                        1. **Where to keep their Emergency Fund:** Name specific examples (e.g., keeping it in an automatic sweep-in Fixed Deposit or a stable Liquid Mutual Fund for instant access).
                        2. **How to start their Wealth Investment:** Explain how to set up a monthly **SIP (Systematic Investment Plan)** via popular user-friendly apps (like Groww, Zerodha Coin, or Kuvera). Suggest what general category of fund fits them (e.g., Nifty 50 Index Funds for beginners vs Conservative Hybrid Funds).
                        3. **Milestone tracking:** Give them a realistic timeline based on their specific reason: "{st.session_state.graph_state['profile'].reason_for_investment}".
                        
                        Keep the tone encouraging, avoid deep complex financial jargon, and focus entirely on action steps a beginner can complete on a smartphone.
                        """
                        detailed_plan = llm.invoke([HumanMessage(content=blueprint_prompt)]).content
                    
                    st.markdown("### 🏁 Final Practical Action Plan")
                    st.markdown(detailed_plan)
                    st.session_state.graph_state["messages"].append(AIMessage(content=f"### 🏁 Final Practical Action Plan\n{detailed_plan}"))
                    st.session_state.graph_state["current_step"] = "COMPLETED"
                else:
                    # Reset steps if the user wants modifications
                    st.session_state.graph_state["current_step"] = "GATHERING"
                    st.session_state.graph_state["initial_recommendation"] = ""
                    reset_msg = "No problem at all! Let's adapt the plan. What changes should we make to your investment amounts or targets?"
                    st.markdown(reset_msg)
                    st.session_state.graph_state["messages"].append(AIMessage(content=reset_msg))
                    
        st.rerun()
else:
    st.success("Plan built successfully! Refresh the page to launch a new session.")