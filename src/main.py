import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama
from typing import Optional
import time

# Streamlit configuration
st.set_page_config(
    page_title="ChatBot by DataDiggerz",
    page_icon="💬",
    layout="wide"
)

class DataFrameChat:
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "df" not in st.session_state:
            st.session_state.df = None
        if "file_name" not in st.session_state:
            st.session_state.file_name = None
        if "llm" not in st.session_state:
            st.session_state.llm = None

    def read_data(self, file) -> Optional[pd.DataFrame]:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            return df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None

    def setup_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Settings")
            model = st.selectbox(
                "Select Model",
                ["mistral", "llama2"],
                index=0
            )
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,  # Set to 0 for more deterministic responses
                step=0.1
            )
            return model, temperature

    def initialize_llm(self, model: str, temperature: float):
        try:
            st.session_state.llm = ChatOllama(
                model=model,
                temperature=temperature,
                timeout=30  # Add timeout for LLM calls
            )
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            return None

    def handle_basic_queries(self, query: str) -> Optional[str]:
        """Handle basic DataFrame queries without using the agent"""
        query_lower = query.lower()
        
        try:
            if "number of rows" in query_lower or "how many rows" in query_lower:
                return f"The DataFrame has {len(st.session_state.df)} rows."
            
            if "number of columns" in query_lower or "how many columns" in query_lower:
                return f"The DataFrame has {len(st.session_state.df.columns)} columns."
            
            if "show columns" in query_lower or "what columns" in query_lower or "list columns" in query_lower:
                return f"The columns are: {', '.join(st.session_state.df.columns)}"
            
            if "show first" in query_lower or "show top" in query_lower:
                n = 5  # default number of rows
                return f"Here are the first {n} rows:\n```\n{st.session_state.df.head(n).to_string()}\n```"
                
            return None
            
        except Exception as e:
            return f"Error processing basic query: {str(e)}"

    def create_agent(self):
        if st.session_state.df is not None and st.session_state.llm is not None:
            try:
                return create_pandas_dataframe_agent(
                    llm=st.session_state.llm,
                    df=st.session_state.df,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=2,  # Reduce max iterations
                    allow_python_repl=True,
                    allow_dangerous_code=True
                )
            except Exception as e:
                st.error(f"Error creating agent: {str(e)}")
                return None
        return None

    def process_query(self, agent, user_prompt: str) -> str:
        # First try handling basic queries
        basic_response = self.handle_basic_queries(user_prompt)
        if basic_response:
            return basic_response
            
        # If not a basic query, use the agent
        try:
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = agent.invoke(user_prompt)
                if time.time() - start_time > 10:  # If taking too long
                    st.warning("Response took longer than expected. Consider rephrasing your question.")
                return str(response['output']) if isinstance(response, dict) else str(response)
        except Exception as e:
            error_msg = str(e)
            if "parsing errors" in error_msg.lower():
                return "I apologize, but I had trouble understanding the output. Could you rephrase your question?"
            return f"I encountered an error: {error_msg}. Please try rephrasing your question."

    def display_data_info(self):
        if st.session_state.df is not None:
            with st.expander("📊 Data Information", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("DataFrame Preview:")
                    st.dataframe(st.session_state.df.head(), use_container_width=True)
                with col2:
                    st.write("Data Statistics:")
                    st.write(f"Total Rows: {len(st.session_state.df)}")
                    st.write(f"Total Columns: {len(st.session_state.df.columns)}")
                    st.write("Columns:", ", ".join(st.session_state.df.columns))

    def handle_file_upload(self):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls"],
            help="Upload your data file (CSV or Excel)"
        )
        
        if uploaded_file and (not st.session_state.file_name or 
                            st.session_state.file_name != uploaded_file.name):
            with st.spinner("Loading data..."):
                st.session_state.df = self.read_data(uploaded_file)
                st.session_state.file_name = uploaded_file.name
                if st.session_state.df is not None:
                    st.success(f"Successfully loaded {uploaded_file.name}")

    def run(self):
        st.title("🤖 Vois ChatBot by DataDiggerz")
        
        model, temperature = self.setup_sidebar()
        self.initialize_llm(model, temperature)
        
        self.handle_file_upload()
        self.display_data_info()
        
        st.divider()
        st.subheader("💬Chat")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_prompt = st.chat_input(
            "Ask about your data...",
            disabled=st.session_state.df is None
        )
        
        if user_prompt:
            st.chat_message("user").markdown(user_prompt)
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_prompt
            })
            
            agent = self.create_agent()
            if agent:
                response = self.process_query(agent, user_prompt)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                with st.chat_message("assistant"):
                    st.markdown(response)
            else:
                st.warning("Please ensure data and model are properly loaded.")

if __name__ == "__main__":
    app = DataFrameChat()
    app.run()