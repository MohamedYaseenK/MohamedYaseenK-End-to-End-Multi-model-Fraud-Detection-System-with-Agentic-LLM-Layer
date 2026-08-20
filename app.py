import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from model import score_transaction
from agent import investigate_transaction

st.set_page_config(page_title="Fraud Detection Demo", layout="centered")
st.title("Fraud Detection + Agentic Investigation")

st.subheader("Transaction")
col1, col2 = st.columns(2)

with col1:
    txn_type = st.selectbox("Type", ["TRANSFER", "CASH_OUT"])
    amount = st.number_input("Amount", min_value=0.0, value=181.0)
    name_orig = st.text_input("nameOrig", value="C1231006815")
    old_bal_org = st.number_input("oldbalanceOrg", min_value=0.0, value=181.0)

with col2:
    name_dest = st.text_input("nameDest", value="C1666544295")
    old_bal_dest = st.number_input("oldbalanceDest", min_value=0.0, value=0.0)
    new_bal_org = st.number_input("newbalanceOrig", min_value=0.0, value=0.0)
    new_bal_dest = st.number_input("newbalanceDest", min_value=0.0, value=0.0)

if st.button("Score Transaction"):
    transaction = {
        "type": txn_type,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_bal_org,
        "newbalanceOrig": new_bal_org,
        "nameDest": name_dest,
        "oldbalanceDest": old_bal_dest,
        "newbalanceDest": new_bal_dest,
    }

    result = score_transaction(transaction)
    st.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")

    if result["is_flagged"]:
        st.error("FLAGGED — running agent investigation...")
        with st.spinner("Agent investigating..."):
            report = investigate_transaction(transaction, result["fraud_probability"])

        st.subheader("Case Report")
        st.markdown(f"**Risk Level:** {report['risk_level']}")
        st.markdown("**Key Indicators:**")
        for indicator in report["key_indicators"]:
            st.markdown(f"- {indicator}")
        st.markdown(f"**Recommended Action:** {report['recommended_action']}")
        st.markdown(f"**Summary:** {report['summary']}")
    else:
        st.success("Not flagged — no investigation needed.")