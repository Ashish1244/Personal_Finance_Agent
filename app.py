import streamlit as st
import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

# ==========================================
# 1. PAGE SETUP & DESIGN STYLING (CSS overrides)
# ==========================================
st.set_page_config(
    page_title="WealthAI | Financial Blueprint Engine",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Premium Dark Mode Theme Styling (Matches your screenshot layout perfectly)
st.markdown("""
    <style>
    /* Set main page background to dark slate matching the screenshot */
    .stApp {
        background-color: #0f172a !important;
        color: #ffffff !important;
    }
    
    /* TRANSFORMS THE CHAT INPUT AREA TO MATCH YOUR DESIGN INTERFACE */
    div[data-testid="stChatInput"] {
        border-radius: 16px !important;
        padding: 10px !important;
        background-color: #1e293b !important; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.3) !important;
        border: 1px solid #334155 !important;
    }
    
    /* Minimal single-line style for the chat input field text area */
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid #475569 !important;
        border-radius: 0px !important;
        color: #ffffff !important;
        box-shadow: none !important;
        padding: 6px 0px !important;
    }
    
    /* Highlight the input underline with electric blue when active */
    div[data-testid="stChatInput"] textarea:focus {
        border-bottom-color: #38bdf8 !important;
    }
    
    /* Font style fixes for typed text and placeholder strings */
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }
    
    /* METRIC CARDS: High contrast visibility */
    .metric-card {
        background-color: #1e293b !important;
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #38bdf8 !important;
        margin-bottom: 15px;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    .metric-card b {
        color: #94a3b8 !important;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .metric-val {
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* 🔥 NEW PREMIUM BLUEPRINT RECOMMENDATION BOX STYLING */
    .blueprint-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin: 20px 0px !important;
        box-shadow: 0 12px 30px rgba(56, 189, 248, 0.15) !important;
    }
    .blueprint-header {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        border-bottom: 1px solid #334155 !important;
        padding-bottom: 14px !important;
        margin-bottom: 18px !important;
    }
    .blueprint-title {
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.02em !important;
    }
    .blueprint-subtitle {
        color: #38bdf8 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 16px !important;
    }
    .blueprint-content {
        color: #cbd5e1 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    /* Style lists beautifully inside the custom card */
    .blueprint-content ul, .blueprint-content ol {
        padding-left: 20px !important;
        margin-bottom: 16px !important;
    }
    .blueprint-content li {
        margin-bottom: 8px !important;
    }
    .blueprint-disclaimer {
        font-size: 0.85rem !important;
        color: #64748b !important;
        border-top: 1px solid #334155 !important;
        padding-top: 12px !important;
        margin-top: 16px !important;
        font-style: italic !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 WealthAI")
st.subheader("Interactive Financial Blueprint Engine")
st.write("Constructing rule-based tactical capital roadmaps securely step-by-step.")
st.markdown("---")

# API Key validation sidebar wrapper
if "GROQ_API_KEY" not in os.environ:
    with st.sidebar:
        st.header("🔑 Authentication Config")
        api_key_input = st.text_input("Enter Groq API Key", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            st.success("API Key applied!")

if not os.environ.get("GROQ_API_KEY"):
    st.warning("🔒 Please set your `GROQ_API_KEY` system environment variable or enter it via the sidebar to start.")
    st.stop()

@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

llm = get_llm()

# ==========================================
# 2. DATA PARSING BLUEPRINT SCHEMAS
# ==========================================
class IncomeParser(BaseModel):
    monthly_income: float = Field(description="Net monthly income value.")

class BaseExpenseParser(BaseModel):
    total_declared_expenses: float = Field(description="Total base monthly expenses.")

class MonthlyExpenseParser(BaseModel):
    total_declared_expenses: float = Field(description="The finalized total monthly expense number.")

class SafetyNetParser(BaseModel):
    has_high_interest_debt: bool = Field(description="True if they have credit card or personal loan debt.")
    has_emergency_fund: bool = Field(description="True if they have savings buffers or active health insurance.")

class InvestorDnaParser(BaseModel):
    risk_profile: str = Field(description="Must be exactly: Conservative, Moderate, or Aggressive.")
    investment_horizon: str = Field(description="Must be exactly: Short-term or Long-term.")

class PlanBApprovalParser(BaseModel):
    is_approved: bool = Field(description="True if they approve Plan B, False if they want changes.")
    feedback_notes: str = Field(description="Minor tweaks requested if not approved.")

# ==========================================
# 3. INITIALIZE STATE VARIABLES
# ==========================================
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
    st.session_state.monthly_income = 0.0
    st.session_state.total_declared_expenses = 0.0
    st.session_state.disposable_income = 0.0
    st.session_state.has_high_interest_debt = None
    st.session_state.has_emergency_fund = None
    st.session_state.risk_profile = "unknown"
    st.session_state.investment_horizon = "unknown"
    st.session_state.plan_a = None
    st.session_state.plan_b = None
    st.session_state.final_approved_plan = None
    st.session_state.agent_logs = []
    st.session_state.last_agent_prompt = "👋 Welcome! Let's build your financial roadmap. To start, what is your total monthly net income (take-home pay)?"

# ==========================================
# 4. STEP MUTATION PIPELINE CONTROLLER
# ==========================================
def run_step_mutation(user_input: str):
    step = st.session_state.current_step
    cleaned_input = user_input.lower().strip()
    
    if "what is my income" in cleaned_input or "my income" in cleaned_input:
        st.session_state.agent_logs.append(("👤 YOU", user_input))
        st.session_state.agent_logs.append(("🤖 AGENT [Memory Callback]", f"Your recorded net monthly income is **${st.session_state.monthly_income:,.2f}**."))
        return

    st.session_state.agent_logs.append(("👤 YOU", user_input))

    # --- STEP 1 ---
    if step == 1:
        parsed = llm.with_structured_output(IncomeParser).invoke(f"Extract number: '{user_input}'")
        st.session_state.monthly_income = parsed.monthly_income
        st.session_state.current_step = 2
        st.session_state.last_agent_prompt = f"Recorded Income: **${st.session_state.monthly_income:,.2f}**.\n\nNext, what is a rough estimate of your total monthly expenses?"
    
    # --- STEP 2 ---
    elif step == 2:
        parsed = llm.with_structured_output(BaseExpenseParser).invoke(f"Extract number: '{user_input}'")
        st.session_state.total_declared_expenses = parsed.total_declared_expenses
        st.session_state.current_step = 3
        st.session_state.last_agent_prompt = f"Got it. So your total estimated month-wise outgoings are **${st.session_state.total_declared_expenses:,.2f}**. Please confirm if this flat monthly figure is correct, or update it if you'd like to adjust the total."

    # --- STEP 3 ---
    elif step == 3:
        parsed = llm.with_structured_output(MonthlyExpenseParser).invoke(f"Extract final baseline total from: '{user_input}'. Fallback: {st.session_state.total_declared_expenses}")
        st.session_state.total_declared_expenses = parsed.total_declared_expenses
        
        # Automated Step 4 execution
        st.session_state.disposable_income = st.session_state.monthly_income - st.session_state.total_declared_expenses
        st.session_state.agent_logs.append(("🤖 AGENT [System Log]", f"📊 Arithmetic Complete!\nDisposable Income Available for Allocation: **${st.session_state.disposable_income:,.2f}**"))
        
        st.session_state.current_step = 5
        st.session_state.last_agent_prompt = "Let's audit your financial safety nets 🛡️:\n1. Do you have high-interest debt like active credit card balances?\n2. Do you have an emergency fund or medical insurance buffer set up?"

    # --- STEP 5 ---
    elif step == 5:
        parsed = llm.with_structured_output(SafetyNetParser).invoke(f"Extract safety status parameters: '{user_input}'")
        st.session_state.has_high_interest_debt = parsed.has_high_interest_debt
        st.session_state.has_emergency_fund = parsed.has_emergency_fund
        st.session_state.current_step = 6
        st.session_state.last_agent_prompt = "Let's map your Investor DNA 🧬:\n1. What is your investment timeline? (Short-term under 2 years vs Long-term wealth building)?\n2. What is your risk comfort level? (Conservative, Moderate, or Aggressive)?"

    # --- STEP 6 ---
    elif step == 6:
        parsed = llm.with_structured_output(InvestorDnaParser).invoke(f"Extract preferences text profiles: '{user_input}'")
        st.session_state.risk_profile = parsed.risk_profile
        st.session_state.investment_horizon = parsed.investment_horizon
        
        # Automated Step 7 strategy engine call
        prompt = f"""Generate a detailed asset allocation strategy called 'Plan A' for this profile:
        Income: ${st.session_state.monthly_income} | Flat Monthly Expenses: ${st.session_state.total_declared_expenses} | Available Capital: ${st.session_state.disposable_income}
        Has Debt: {st.session_state.has_high_interest_debt} | Has Emergency Fund: {st.session_state.has_emergency_fund}
        Investor DNA: {st.session_state.risk_profile} risk profile targeting a {st.session_state.investment_horizon} timeline.
        Provide exact budget splits using real dollar numbers. Do not include markdown formatting or disclaimer notes in the raw text generation."""
        
        st.session_state.plan_a = llm.invoke(prompt).content
        st.session_state.current_step = 8
        st.session_state.last_agent_prompt = "Does this 'Plan A' breakdown look realistic for you, or would you like to tweak the balance?"

    # --- STEP 8 ---
    elif step == 8:
        summary = llm.invoke(f"Summarize the user's specific complaints or tweaks requested: '{user_input}'").content
        st.session_state.plan_b = summary
        
        # Automated Step 9 rebalancing call
        prompt = f"""The user rejected Plan A. Create a revised allocation strategy called 'Plan B' that resolves their exact issue.
        Original Plan A: {st.session_state.plan_a}
        User's Objection: "{st.session_state.plan_b}"
        Total Allocation Capital: ${st.session_state.disposable_income:.2f}
        Output a revised allocation budget explaining how their objection was addressed. Do not include markdown formatting titles or disclaimer notes in the raw text generation."""
        
        st.session_state.plan_b = llm.invoke(prompt).content
        st.session_state.current_step = 10
        st.session_state.last_agent_prompt = "Does 'Plan B' meet your expectations? Are you ready to lock it in?"

    # --- STEP 10 ---
    elif step == 10:
        parsed = llm.with_structured_output(PlanBApprovalParser).invoke(f"Is user approving? Response: '{user_input}'")
        if parsed.is_approved:
            st.session_state.final_approved_plan = st.session_state.plan_b
            
            # Automated Step 11 simulation output save
            st.session_state.agent_logs.append(("📡 SYSTEM", "⚠️ [DB UPSERT] Safely committing metrics and 'final_approved_plan' to your Database profile..."))
            st.session_state.current_step = -1
            st.session_state.last_agent_prompt = "🎉 Strategy Saved Successfully!\n\nYour long-term financial roadmap is now locked in. Go follow the tactical allocation roadmap targets. This active chat session is closed."
        else:
            st.session_state.plan_b = parsed.feedback_notes
            
            prompt = f"""Re-optimize the financial plan using the new constraints:
            Original: {st.session_state.plan_a}
            New Feedback: "{st.session_state.plan_b}"
            Capital: ${st.session_state.disposable_income:.2f}"""
            
            st.session_state.plan_b = llm.invoke(prompt).content
            st.session_state.current_step = 10
            st.session_state.last_agent_prompt = "I have updated the allocation ratios based on that feedback. Does this option align better?"

# ==========================================
# 5. UI DISPLAY & RENDERING ENGINE LAYER
# ==========================================

# Financial Metrics Panel Row (Styled with Dark Mode High-Contrast Cards)
if st.session_state.monthly_income > 0:
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"<div class='metric-card'><b>Income</b><div class='metric-val'>${st.session_state.monthly_income:,.2f}</div></div>", unsafe_allow_html=True)
    with cols[1]:
        if st.session_state.total_declared_expenses > 0:
            st.markdown(f"<div class='metric-card'><b>Expenses</b><div class='metric-val'>${st.session_state.total_declared_expenses:,.2f}</div></div>", unsafe_allow_html=True)
    with cols[2]:
        if st.session_state.disposable_income > 0:
            st.markdown(f"<div class='metric-card' style='border-left-color:#10b981 !important;'><b>Disposable</b><div class='metric-val'>${st.session_state.disposable_income:,.2f}</div></div>", unsafe_allow_html=True)

# Main conversation logger mapping panel
for sender, message in st.session_state.agent_logs:
    if "👤 YOU" in sender:
        st.markdown(f"**{sender}:** {message}")
    elif "Memory Callback" in sender:
        st.info(message)
    elif "System Log" in sender or "DB UPSERT" in message:
        st.caption(message)
    else:
        st.markdown(f"**{sender}:** {message}")

# 🔥 BEAUTIFUL RENDER FOR GENERATED BLUEPRINTS
if st.session_state.current_step in [8, 9, 10] and st.session_state.plan_a:
    import markdown
    html_plan_a = markdown.markdown(st.session_state.plan_a)
    st.markdown(f"""
        <div class="blueprint-container">
            <div class="blueprint-header">
                <span style="font-size: 1.6rem;">📊</span>
                <h3 class="blueprint-title">Strategic Plan A Blueprint</h3>
            </div>
            <div class="blueprint-subtitle">Initial Target Matrix Allocation</div>
            <div class="blueprint-content">
                {html_plan_a}
            </div>
            <div class="blueprint-disclaimer">
                ⚠️ REGULATORY DISCLAIMER: Visualization model framework target parameters only. Not certified financial investment planning advice.
            </div>
        </div>
    """, unsafe_allow_html=True)

if st.session_state.current_step == 10 and st.session_state.plan_b:
    import markdown
    html_plan_b = markdown.markdown(st.session_state.plan_b)
    st.markdown(f"""
        <div class="blueprint-container" style="border-color: #10b981 !important;">
            <div class="blueprint-header" style="border-bottom-color: #064e3b !important;">
                <span style="font-size: 1.6rem;">🔄</span>
                <h3 class="blueprint-title" style="color: #ffffff;">Revised Strategic Plan B Blueprint</h3>
            </div>
            <div class="blueprint-subtitle" style="color: #10b981 !important;">Optimized Rebalanced Configuration Matrix</div>
            <div class="blueprint-content">
                {html_plan_b}
            </div>
            <div class="blueprint-disclaimer">
                ⚠️ REGULATORY DISCLAIMER: Visualization model framework target parameters only. Not certified financial investment planning advice.
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Active processing element interface
if st.session_state.current_step != -1:
    st.markdown("### 🤖 Agent Query")
    st.info(st.session_state.last_agent_prompt)

    # STREAMLIT CHAT INPUT: Locks keyword focus automatically on enter keys
    user_response = st.chat_input("Type your response or ask 'what is my income'...")
    if user_response and user_response.strip():
        run_step_mutation(user_response)
        st.rerun()
else:
    st.markdown("### 🏁 Final Execution Complete")
    st.success(st.session_state.last_agent_prompt)
    
    with st.expander("💼 View Final Approved Saved Strategy Blueprint"):
        st.markdown(st.session_state.final_approved_plan)
        
    if st.button("🔄 Restart Allocation Analysis Engine"):
        st.session_state.clear()
        st.rerun()