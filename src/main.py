import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from typing import Optional
import time
from dotenv import load_dotenv
from langchain.agents import AgentExecutor

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        if "sheet_names" not in st.session_state:
            st.session_state.sheet_names = []
        if "selected_sheet" not in st.session_state:
            st.session_state.selected_sheet = None
        if "last_displayed_data" not in st.session_state:
            st.session_state.last_displayed_data = None
        if "reserve_row_index" not in st.session_state:
            st.session_state.reserve_row_index = None
        if "reservation_mode" not in st.session_state:
            st.session_state.reservation_mode = False
        if "show_success" not in st.session_state:
            st.session_state.show_success = False
        if "success_message" not in st.session_state:
            st.session_state.success_message = ""

    def read_data(self, file) -> Optional[pd.DataFrame]:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                st.session_state.sheet_names = ["Sheet1"]
                st.session_state.selected_sheet = "Sheet1"
                return df
            elif file.name.endswith(".xlsx") or file.name.endswith(".xls"):
                xls = pd.ExcelFile(file, engine='openpyxl')
                st.session_state.sheet_names = xls.sheet_names
                if not st.session_state.selected_sheet:
                    st.session_state.selected_sheet = st.session_state.sheet_names[0]
                df = pd.read_excel(file, sheet_name=st.session_state.selected_sheet, engine='openpyxl')
                return df
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None

    def setup_sidebar(self):
        model = "gpt-4"
        temperature = 0.5
        return model, temperature
        #with st.sidebar:
            #st.header("⚙️ Settings")
            #model = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
            
            #temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            # if st.session_state.sheet_names:
            #     st.session_state.selected_sheet = st.selectbox(
            #         "Select Sheet",
            #         st.session_state.sheet_names,
            #         index=st.session_state.sheet_names.index(st.session_state.selected_sheet))
           

    def initialize_llm(self, model: str, temperature: float):
        try:
            st.session_state.llm = ChatOpenAI(model_name=model, openai_api_key=OPENAI_API_KEY, temperature=temperature)
            return st.session_state.llm
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            return None

    def create_agent(self):
        if st.session_state.df is not None and st.session_state.llm is not None:
            try:
                agent = create_pandas_dataframe_agent(
                    llm=st.session_state.llm,
                    df=st.session_state.df,
                    verbose=True,
                    max_iterations=2,
                    allow_dangerous_code=True
                )
                return AgentExecutor.from_agent_and_tools(
                    agent=agent.agent,
                    tools=agent.tools,
                    handle_parsing_errors=True
                )
            except Exception as e:
                st.error(f"Error creating agent: {str(e)}")
                return None
        return None

    def process_query(self, agent, user_prompt: str) -> str:
        try:
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = agent.invoke(user_prompt)
                if time.time() - start_time > 10:
                    st.warning("Response took longer than expected. Consider rephrasing your question.")
                return str(response['output']) if isinstance(response, dict) else str(response)
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try rephrasing your question."

    def handle_file_upload(self):
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"], help="Upload your data file (CSV or Excel)")
        
        if uploaded_file and (not st.session_state.file_name or st.session_state.file_name != uploaded_file.name):
            with st.spinner("Loading data..."):
                st.session_state.df = self.read_data(uploaded_file)
                st.session_state.file_name = uploaded_file.name
                if st.session_state.df is not None:
                    st.success(f"Successfully loaded {uploaded_file.name} (Sheet: {st.session_state.selected_sheet})")
                    st.session_state.last_displayed_data = st.session_state.df.head()
                    st.write(st.session_state.last_displayed_data)

    def add_reserved_column(self):
        """Add a 'Reserved By' column to the DataFrame if it doesn't already exist."""
        if st.session_state.df is not None and "Reserved By" not in st.session_state.df.columns:
            st.session_state.df["Reserved By"] = pd.NA

    def reserve_data(self, row_index: int, reserved_by: str = "Test User") -> bool:
        """Reserve a row by updating the 'Reserved By' column."""
        if st.session_state.df is not None and row_index < len(st.session_state.df):
            try:
                current_value = st.session_state.df.at[row_index, "Reserved By"]
                if pd.isna(current_value):
                    st.session_state.df.at[row_index, "Reserved By"] = reserved_by
                    st.session_state.show_success = True
                    st.session_state.success_message = f"✅ Row {row_index} has been successfully reserved by {reserved_by}."
                    return True
                else:
                    st.warning(f"⚠️ Row {row_index} is already reserved by {current_value}")
                    return False
            except Exception as e:
                st.error(f"❌ Error during reservation: {str(e)}")
                return False
        else:
            st.error("❌ Invalid row index or no data loaded.")
            return False

    def run(self):
        st.title("🤖 Vois ChatBot by DataDiggerz")
        model, temperature = self.setup_sidebar()
        self.initialize_llm(model, temperature)
        self.handle_file_upload()
        
        # Add 'Reserved By' column if it doesn't exist
        self.add_reserved_column()
        
        # Display success message if it exists
        if st.session_state.show_success:
            st.success(st.session_state.success_message)
            st.session_state.show_success = False  # Reset the flag after displaying the message
        
        st.subheader("💬 Chat")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_prompt = st.chat_input("Ask about your data...", disabled=st.session_state.df is None)
        
        if user_prompt:
            st.chat_message("user").markdown(user_prompt)
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            
            # Check if the user wants to reserve data
            if "reserve" in user_prompt.lower():
                st.session_state.reservation_mode = True
                
            if st.session_state.reservation_mode:
                if st.session_state.df is not None:
                    st.write("Current data:")
                    st.write(st.session_state.df)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        row_index = st.number_input(
                            "Enter the row index to reserve:",
                            min_value=0,
                            max_value=len(st.session_state.df) - 1,
                            value=0,
                            step=1
                        )
                    with col2:
                        if st.button("Confirm Reservation"):
                            if self.reserve_data(row_index):
                                # Add successful reservation to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": st.session_state.success_message
                                })
                                st.session_state.reservation_mode = False
                                # Re-render the DataFrame to show the updated reservation
                                st.write("Updated data:")
                                st.write(st.session_state.df)
            else:
                # Process other queries using the agent
                agent = self.create_agent()
                if agent:
                    response = self.process_query(agent, user_prompt)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                else:
                    response = "Please ensure data and model are properly loaded."
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)

if __name__ == "__main__":
    app = DataFrameChat()
    app.run()