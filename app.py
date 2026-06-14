import os
import streamlit as st
from typing import Annotated, Literal
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. CORE AGENT LOGIC & DATA SCHEMAS
# ==========================================

class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    step: str = "income"
    income: float = 0.0
    expenses: float = 0.0
    risk_appetite: str = "Low"

class FinancialAdjustments(BaseModel):
    reasoning: str = Field(description="Brief explanation of what the user wants to change.")
    income_update: float = Field(description="The new monthly income value if explicitly changed. Otherwise, return current income exactly.")
    expenses_update: float = Field(description="The new monthly expenses value if explicitly changed. Otherwise, return current expenses exactly.")
    risk_update: Literal["High", "Low", "Toggle", "No_Change"] = Field(description="Select 'High' or 'Low' if specified. CRITICAL: If the user explicitly asks for another option, a different plan, an alternative, or says they dislike the current setup, you MUST choose 'Toggle'.")

def generate_financial_plan(income: float, expenses: float, risk: str) -> str:
    """Outputs a clean structural format for the Streamlit UI parser to read."""
    surplus = income - expenses
    if surplus <= 0:
        return "⚠️ BUDGET_ALERT|Your spending matches or exceeds your income. Prioritize setting up a 3-month basic **Emergency Fund** before investing."
    
    # Fully distinct product recommendation allocation templates
    if risk == "Low":
        return (
            f"📊 PLAN_DATA|{surplus:.2f}|"
            f"🔒 Debt Mutual Funds & Fixed Deposits (50%) + 🏥 Term & Health Insurance Premium (20%)|"
            f"🚨 Emergency Fund Liquid Savings Buffer (30%)|"
            f"Low Risk & Capital Preservation"
        )
    else:
        return (
            f"📊 PLAN_DATA|{surplus:.2f}|"
            f"🚀 High-Growth Equity Mutual Funds & Index SIPs (70%)|"
            f"🎯 Multi-Cap Funds (10%) + 🚨 Emergency Fund / Insurance (20%)|"
            f"Aggressive Long-Term Wealth Compounding"
        )

def finance_agent_node(state: AgentState):
    user_input = state.messages[-1].content if state.messages else ""
    updated_step = state.step
    updated_income = state.income
    updated_expenses = state.expenses
    updated_risk = state.risk_appetite
    
    api_key = os.environ.get("GROQ_API_KEY") or st.session_state.get("groq_api_key")
    if not api_key:
        return {"messages": [AIMessage(content="Please provide a valid Groq API Key in the sidebar to begin.")], "step": updated_step}
        
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=api_key)
    
    # --- STAGE 1: Extract Income ---
    if updated_step == "income":
        res = llm.invoke(f"Extract only the total monthly income number from: '{user_input}'. Return just numbers.").content
        try:
            updated_income = float(''.join(c for c in res if c.isdigit() or c=='.'))
            updated_step = "expenses"
            response = "Got it! Now, what are your approximate monthly expenses? (e.g., rent, bills, subscriptions, coffee, travel, etc.)"
        except ValueError:
            response = "I couldn't quite catch the amount. Could you please state your monthly income clearly in digits?"
            
    # --- STAGE 2: Extract Expenses ---
    elif updated_step == "expenses":
        res = llm.invoke(f"Extract only the total monthly expense number from: '{user_input}'. Return just numbers.").content
        try:
            updated_expenses = float(''.join(c for c in res if c.isdigit() or c=='.'))
            updated_step = "clarifying"
            response = "Thanks. To customize your plan, how do you feel about investment risk? Would you prefer Low risk (safe, stable) or High risk (higher growth, volatile)?"
        except ValueError:
            response = "Could you please give me a rough estimate of your total monthly expenses as a number?"

    # --- STAGE 3: Extract Risk & Present Recommendation ---
    elif updated_step == "clarifying":
        res = llm.invoke(f"Is this 'Low' or 'High' risk appetite?: '{user_input}'. Reply with just one word.").content.strip()
        updated_risk = "High" if "High" in res else "Low"
        updated_step = "review_recommendation"
        response = generate_financial_plan(updated_income, updated_expenses, updated_risk)

    # --- STAGE 4: Loop Confirmation ---
    elif updated_step == "review_recommendation":
        confirmation_prompt = (
            f"Analyze if the user is satisfied and confirming the plan, or if they want changes/dislike it. "
            f"User input: '{user_input}'. Reply with exactly 'CONFIRMED' if they say yes, look good, agree, or thank you. Otherwise, reply 'CHANGES_REQUESTED'."
        )
        validation = llm.invoke(confirmation_prompt).content.strip()
        
        if "CONFIRMED" in validation:
            updated_step = "done"
            response = "Wonderful! I'm glad you found the plan helpful. Feel free to come back if your financial situation changes. Happy investing! 🚀"
        else:
            structured_llm = llm.with_structured_output(FinancialAdjustments)
            extraction_context = (
                f"The user wants changes or a completely alternative strategy. User Feedback: '{user_input}'. "
                f"Current Profile -> Income: {state.income}, Expenses: {state.expenses}, Risk Profile: {state.risk_appetite}.\n\n"
                f"If they state they dislike the approach, or want to see something else, make sure to mark risk_update as 'Toggle'."
            )
            
            try:
                result: FinancialAdjustments = structured_llm.invoke(extraction_context)
                
                income_changed = result.income_update != state.income
                expenses_changed = result.expenses_update != state.expenses
                
                updated_income = result.income_update
                updated_expenses = result.expenses_update
                
                # 🔥 FIX: Read and toggle against 'updated_risk' variable, not the stale 'state.risk_appetite'
                if result.risk_update == "Toggle":
                    updated_risk = "Low" if updated_risk == "High" else "High"
                elif result.risk_update in ["High", "Low"]:
                    updated_risk = result.risk_update

                if income_changed or expenses_changed:
                    updated_step = "clarifying"
                    response = (
                        f"Understood. I have updated your monthly income to **${updated_income:,.2f}** "
                        f"and expenses to **${updated_expenses:,.2f}**. "
                        f"Before I regenerate the strategy, do you want to keep your risk appetite as **{updated_risk} Risk**, or switch it?"
                    )
                else:
                    context_prompt = (
                        f"You are a personal finance assistant. You have successfully updated their profile parameters to -> "
                        f"Income: {updated_income}, Expenses: {updated_expenses}, Risk Appetite: {updated_risk}.\n"
                        f"User input was: '{user_input}'. Write a 1-sentence friendly confirmation stating you have changed their strategy below."
                    )
                    conversational_ack = llm.invoke(context_prompt).content
                    
                    # 🔥 FIX: Pass 'updated_risk' to generate the completely updated view layout
                    response = f"{conversational_ack}\n\n" + generate_financial_plan(updated_income, updated_expenses, updated_risk)
                    
            except Exception:
                if "other" in user_input.lower() or "another" in user_input.lower() or "change" in user_input.lower():
                    updated_risk = "Low" if updated_risk == "High" else "High"
                response = "I've refreshed your financial profile updates! " + generate_financial_plan(updated_income, updated_expenses, updated_risk)

    return {
        "messages": [AIMessage(content=response)],
        "step": updated_step,
        "income": updated_income,
        "expenses": updated_expenses,
        "risk_appetite": updated_risk
    }

@st.cache_resource
def compile_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("finance_agent", finance_agent_node)
    workflow.add_edge(START, "finance_agent")
    workflow.add_edge("finance_agent", END)
    return workflow.compile(checkpointer=MemorySaver())

app = compile_workflow()

# ==========================================
# 2. STREAMLIT FRONTEND & UI INTERFACE
# ==========================================

st.set_page_config(page_title="WealthWise Agent", page_icon="💰", layout="centered")

with st.sidebar:
    st.title("⚙️ Control Settings")
    st.markdown("Configure your environment keys and settings below.")
    
    if "GROQ_API_KEY" in os.environ:
        st.success("Groq API Key detected from system environment.")
        st.session_state.groq_api_key = os.environ["GROQ_API_KEY"]
    else:
        user_key = st.text_input("Enter Groq API Key", type="password")
        if user_key:
            st.session_state.groq_api_key = user_key
            st.success("API Key saved for session!")
            
    st.divider()
    st.markdown("### 🤖 Architecture Details")
    st.info("**Engine:** LangGraph FSM\n\n**LLM Model:** Llama 3.3 70B Versatile\n\n**Data Rules:** Dynamic Risk Allocation Engine")
    
    if st.button("🔄 Reset Chat Session"):
        if "chat_history" in st.session_state:
            del st.session_state.chat_history
        st.rerun()

st.title("💰 WealthWise Personal Finance Agent")
st.markdown("Structure your monthly financial plan and investments interactively.")
st.divider()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! Let's map out your investments. What is your total monthly income?"}
    ]

config = {"configurable": {"thread_id": "streamlit_session_v5"}}

def render_beautiful_plan(raw_string):
    """Parses structural template tokens and presents high-end visual cards."""
    if "\n\n📊 PLAN_DATA|" in raw_string:
        header, data_part = raw_string.split("\n\n📊 PLAN_DATA|")
        st.markdown(header)
        parts = ["📊 PLAN_DATA"] + data_part.split("|")
    else:
        parts = raw_string.split("|")
        
    if "BUDGET_ALERT" in parts[0] or "⚠️" in parts[0]:
        st.warning(parts[0] if len(parts) == 1 else parts[1])
        return

    surplus = parts[1]
    core_strategy = parts[2]
    secondary_strategy = parts[3]
    focus_tag = parts[4].split("\n\n")[0]

    st.success("### 📊 Your Tailored Investment Strategy")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Calculated Monthly Surplus", value=f"${float(surplus):,.2f}")
    with col2:
        st.metric(label="Risk Strategy Profile", value=focus_tag)
        
    st.markdown("#### **Allocation Distribution**")
    st.info(f"🔑 **Primary Focus:** {core_strategy}")
    st.markdown(f"🛡️ **Secondary/Safety Focus:** {secondary_strategy}")
    st.markdown("---")
    st.caption("Does this plan look good to you, or would you like to see the other alternative? (e.g., *'give me another recommendation'*, *'actually look at low risk'*)")

# Display historical messages smoothly
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if "PLAN_DATA|" in message["content"] or "BUDGET_ALERT|" in message["content"]:
            render_beautiful_plan(message["content"])
        else:
            st.markdown(message["content"])

# Chat box trigger
if user_query := st.chat_input("Type your response here..."):
    if not st.session_state.get("groq_api_key"):
        st.error("Please add your Groq API key in the sidebar menu to proceed.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Process input through state machine graph
        output = app.invoke(
            {"messages": [HumanMessage(content=user_query)]}, 
            config=config
        )
        
        latest_msg = output['messages'][-1].content
        
        with st.chat_message("assistant"):
            if "PLAN_DATA|" in latest_msg or "BUDGET_ALERT|" in latest_msg:
                render_beautiful_plan(latest_msg)
            else:
                st.markdown(latest_msg)
                
        st.session_state.chat_history.append({"role": "assistant", "content": latest_msg})
        
        if output["step"] == "done":
            st.balloons()