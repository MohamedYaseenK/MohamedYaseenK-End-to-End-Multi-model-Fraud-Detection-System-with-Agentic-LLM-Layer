from langchain.agents import create_agent
from langchain_core.tools import tool

from agent_tools import get_user_history, get_user_profile, write_case_report

SYSTEM_PROMPT = """You are a fraud investigation agent. A transaction has
already been flagged by an XGBoost classifier as high fraud risk.

Investigate the account using the available tools, then call
write_case_report with your findings.

Steps:
1. Call get_user_history_tool to see the account's recent transactions.
2. Call get_user_profile_tool to see the account's overall summary stats.
3. Reason about whether this transaction is consistent with the
   account's normal behavior.
4. Call write_case_report_tool with risk_level (LOW/MEDIUM/HIGH),
   key_indicators, recommended_action, and a short summary.

Always call write_case_report_tool as your final step."""


@tool
def get_user_history_tool(name_orig: str) -> list[dict]:
    """Get this user's most recent transactions."""
    return get_user_history(name_orig)


@tool
def get_user_profile_tool(name_orig: str) -> dict:
    """Get a summary profile for this user (transaction count, avg amount, prior fraud)."""
    return get_user_profile(name_orig)


@tool
def write_case_report_tool(
    risk_level: str,
    key_indicators: list[str],
    recommended_action: str,
    summary: str,
) -> dict:
    """Write the final structured fraud case report."""
    return write_case_report(risk_level, key_indicators, recommended_action, summary)


TOOLS = [get_user_history_tool, get_user_profile_tool, write_case_report_tool]


def build_agent():
    return create_agent(
        model="google_genai:gemini-3.6-flash",
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


def investigate_transaction(transaction: dict, fraud_probability: float) -> dict:
    agent = build_agent()

    task = (
        f"Investigate this flagged transaction:\n"
        f"nameOrig: {transaction['nameOrig']}\n"
        f"type: {transaction['type']}\n"
        f"amount: {transaction['amount']}\n"
        f"fraud_probability: {fraud_probability}\n"
    )

    result = agent.invoke({"messages": [{"role": "user", "content": task}]})

    for message in reversed(result["messages"]):
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                if call["name"] == "write_case_report_tool":
                    return write_case_report(**call["args"])

    return {
        "risk_level": "UNKNOWN",
        "key_indicators": [],
        "recommended_action": "Agent did not produce a structured report.",
        "summary": result["messages"][-1].content,
    }