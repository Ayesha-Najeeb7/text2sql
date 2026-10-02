"""Streamlit front end:  streamlit run app/app.py   (run from the repository root)"""
import os
import streamlit as st
from inference import generate_sql, ROOT

st.set_page_config(page_title="Text-to-SQL", page_icon="🗄️")
st.title("Text-to-SQL Transformer")
st.caption("Ask a question about a table in plain English. The model — a Transformer trained from scratch on WikiSQL — writes the SQL.")

EXAMPLES = {
    "Toronto Raptors roster": ("What is Terrence Ross' nationality?",
                               "Player, No., Nationality, Position, Years in Toronto, School/Club Team"),
    "Football results": ("How many games were played at home in 2010?", "Year, Opponent, Venue, Score, Attendance"),
}
choice = st.selectbox("Try an example", ["(none)"] + list(EXAMPLES))
q0, c0 = EXAMPLES.get(choice, ("", ""))
question = st.text_input("Question", q0)
cols = st.text_input("Column names (comma-separated)", c0)

if st.button("Generate SQL", type="primary"):
    columns = [c.strip() for c in cols.split(",") if c.strip()]
    if not question.strip() or not columns:
        st.warning("Please enter a question and at least one column name.")
    elif not (ROOT / "checkpoints" / "best.pt").exists():
        st.error("No trained model found. Run train.py first (see README) and put checkpoints/best.pt in place.")
    elif len(columns) > 64:
        st.error("At most 64 columns are supported.")
    else:
        sql, raw, parsed = generate_sql(question, columns)
        if sql is None:
            st.error("The model produced something that is not a valid query for these columns. Try rephrasing.")
        else:
            st.code(sql, language="sql")
        with st.expander("Model output"):
            st.write(raw); st.json(parsed)
