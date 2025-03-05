import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from typing import Optional
import time
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit configuration
st.set_page_config(
    page_title="V-TRAIN",
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
            if isinstance(file, str):
                if file.endswith(".csv"):
                    df = pd.read_csv(file)
                    st.session_state.sheet_names = ["Sheet1"]
                    st.session_state.selected_sheet = "Sheet1"
                elif file.endswith(".xlsx") or file.endswith(".xls"):
                    xls = pd.ExcelFile(file, engine='openpyxl')
                    st.session_state.sheet_names = xls.sheet_names
                    if not st.session_state.selected_sheet:
                        st.session_state.selected_sheet = st.session_state.sheet_names[0]
                    df = pd.read_excel(file, sheet_name=st.session_state.selected_sheet, engine='openpyxl')
                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")
                    return None
            elif hasattr(file, "name"):
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                    st.session_state.sheet_names = ["Sheet1"]
                    st.session_state.selected_sheet = "Sheet1"
                elif file.name.endswith(".xlsx") or file.name.endswith(".xls"):
                    xls = pd.ExcelFile(file, engine='openpyxl')
                    st.session_state.sheet_names = xls.sheet_names
                    if not st.session_state.selected_sheet:
                        st.session_state.selected_sheet = st.session_state.sheet_names[0]
                    df = pd.read_excel(file, sheet_name=st.session_state.selected_sheet, engine='openpyxl')
                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")
                    return None
            else:
                st.error("Invalid file type. Please provide a valid file path or file upload object.")
                return None

            if "Date (mm/dd/yyyy)" in df.columns:
                df["Date (mm/dd/yyyy)"] = pd.to_datetime(df["Date (mm/dd/yyyy)"], errors='coerce')
#convert the customer Id to integer
            if "Customer ID" in df.columns:
                df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors='coerce').fillna(0).astype(int)

            return df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None

    def setup_sidebar(self):
        model = "gpt-4"
        temperature = 0.5
        return model, temperature

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
                # Ensure Customer ID is returned as an integer
                if "customer id" in user_prompt.lower() and isinstance(response, dict):
                    output_text = response.get('output', '')
                    customer_ids = re.findall(r'\d+', output_text)  # Extract numbers
                    if customer_ids:
                        return ", ".join(map(str, map(int, customer_ids)))  # Convert to int & return as string
                return str(response['output']) if isinstance(response, dict) else str(response)
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try rephrasing your question."

    def handle_file_upload(self):
        file_path = "https://raw.githubusercontent.com/yelsangeakshay/chatbot/main/src/chattest.xlsx"
        st.session_state.df = self.read_data(file_path)
        if st.session_state.df is not None:
                st.session_state.last_displayed_data = st.session_state.df.head()
                #st.write(st.session_state.last_displayed_data)

    def add_reserved_column(self):
        if st.session_state.df is not None and "Reserved By" not in st.session_state.df.columns:
            st.session_state.df["Reserved By"] = pd.NA

    def extract_row_number(self, prompt):
        pattern = r'(?:can you )?reserve\s+(?:this\s+)?ro(?:w)?\s*(\d+)'
        match = re.search(pattern, prompt.lower())
        if match:
            return int(match.group(1))
        return None

    def reserve_data(self, row_index: int, reserved_by: str = "Akshay") -> bool:
        if st.session_state.df is not None and 0 <= row_index < len(st.session_state.df):
            try:
                if "Reserved By" not in st.session_state.df.columns:
                    st.session_state.df["Reserved By"] = pd.NA
                
                current_value = st.session_state.df.at[row_index, "Reserved By"]
                if pd.isna(current_value) or current_value == "" or current_value is None:
                    st.session_state.df.at[row_index, "Reserved By"] = reserved_by
                    success_message = f"✅ Row {row_index} has been successfully reserved by {reserved_by}."
                    st.session_state.df = self.clean_dataframe(st.session_state.df)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": success_message
                    })
                    st.success(success_message)
                    return True
                else:
                    failure_message = f"⚠️ Row {row_index} is already reserved by {current_value}"
                    jira_link = "https://yourcompany.atlassian.net/create/ticket"
                    full_message = f"{failure_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": full_message
                    })
                    st.warning(failure_message)
                    st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
                    return False
            except Exception as e:
                failure_message = f"❌ Error during reservation: {str(e)}"
                jira_link = "https://yourcompany.atlassian.net/create/ticket"
                full_message = f"{failure_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": full_message
                })
                st.error(failure_message)
                st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
                return False
        else:
            failure_message = "❌ Invalid row index or no data loaded."
            jira_link = "https://yourcompany.atlassian.net/create/ticket"
            full_message = f"{failure_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_message
            })
            st.error(failure_message)
            st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
            return False

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
        return df

    def run(self):
        st.title("🤖 Vois ChatBot by DataDiggerz")
        model, temperature = self.setup_sidebar()
        self.initialize_llm(model, temperature)
        self.handle_file_upload()
        
        self.add_reserved_column()
        
        st.subheader("💬 Chat")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_prompt = st.chat_input("Ask about your data...", disabled=st.session_state.df is None)
        
        if user_prompt:
            st.chat_message("user").markdown(user_prompt)
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            
            row_number = self.extract_row_number(user_prompt)
            
            if row_number is not None:
                self.reserve_data(row_number)
                st.write("Updated data:")
                st.write(st.session_state.df)
            elif "reserve" in user_prompt.lower():
                response = "Please specify a row number to reserve (e.g., 'can you reserve this row 5')"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            else:
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
