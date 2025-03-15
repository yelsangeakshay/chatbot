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
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False

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
                return str(response['output']) if isinstance(response, dict) else str(response)
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try rephrasing your prompt."

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

    def extract_column_value_reservation(self, prompt):
        pattern = r'(?:can you )?reserve\s+(?:this\s+)?data\s+where\s+(.+?)\s*(?:equals|=)\s*(.+?)(?:\s+|$)' 
        match = re.search(pattern, prompt.lower())
        if match:
            column_name = match.group(1).strip()
            value = match.group(2).strip().strip("'\"").replace(",", "")
            try:
                value = str(int(float(value)))
            except ValueError:
                pass
            return column_name, value
        return None, None

    def extract_with_pattern_reservation(self, prompt):
        pattern = r'(?:can you )?reserve\s+(?:the\s+)?data\s+with\s+(.+?)\s*-\s*(.+?)(?:\s+|$)' 
        match = re.search(pattern, prompt.lower())
        if match:
            column_name = match.group(1).strip()
            value = match.group(2).strip().strip("'\"").replace(",", "")
            try:
                value = str(int(float(value)))
            except ValueError:
                pass
            if column_name.lower() == "customer":
                column_name = "Customer Id"
            return column_name, value
        return None, None

    def find_column_name(self, search_name: str) -> Optional[str]:
        if st.session_state.df is not None:
            for col in st.session_state.df.columns:
                if col.lower().strip() == search_name.lower().strip():
                    return col
        return None

    def reserve_data(self, row_index: int, reserved_by: str = "Akshay") -> bool:
        jira_link = "https://yourcompany.atlassian.net/create/ticket"
        if st.session_state.df is None:
            failure_message = "❌ No data available to reserve."
            full_message = f"{failure_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_message
            })
            st.error(failure_message)
            st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
            return False
            
        if 0 <= row_index < len(st.session_state.df):
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
                full_message = f"{failure_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": full_message
                })
                st.error(failure_message)
                st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
                return False
        else:
            failure_message = "❌ Invalid row index."
            full_message = f"{failure_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_message
            })
            st.error(failure_message)
            st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
            return False

    def reserve_by_column_value(self, column_name: str, value: str, reserved_by: str = "Akshay", wildcard: bool = False) -> bool:
        jira_link = "https://yourcompany.atlassian.net/create/ticket"
        if st.session_state.df is None:
            failure_message = "❌ No data available to reserve."
            full_message = f"{failure_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_message
            })
            st.error(failure_message)
            st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
            return False
            
        actual_column_name = self.find_column_name(column_name)
        if actual_column_name is not None:
            try:
                if "Reserved By" not in st.session_state.df.columns:
                    st.session_state.df["Reserved By"] = pd.NA
                
                if wildcard:
                    mask = st.session_state.df[actual_column_name].notna() & (st.session_state.df[actual_column_name] != "")
                else:
                    df_values = pd.to_numeric(st.session_state.df[actual_column_name], errors='coerce').fillna(st.session_state.df[actual_column_name])
                    df_values = df_values.astype(str).str.replace(r'\.0$', '', regex=True)
                    mask = df_values.str.lower() == str(value).lower()
                
                matching_indices = st.session_state.df[mask].index
                
                if len(matching_indices) == 0:
                    failure_message = f"❌ No rows found where {actual_column_name} has {'any value' if wildcard else f'value {value}'}"
                    full_message = f"{failure_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
                    st.session_state.chat_history.append({"role": "assistant", "content": full_message})
                    st.error(failure_message)
                    st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
                    return False
                
                reserved_count = 0
                already_reserved = []
                
                for idx in matching_indices:
                    current_value = st.session_state.df.at[idx, "Reserved By"]
                    if pd.isna(current_value) or current_value == "" or current_value is None:
                        st.session_state.df.at[idx, "Reserved By"] = reserved_by
                        reserved_count += 1
                    else:
                        already_reserved.append((idx, current_value))
                
                st.session_state.df = self.clean_dataframe(st.session_state.df)
                
                if reserved_count > 0:
                    success_message = f"✅ Successfully reserved {reserved_count} row(s) where {actual_column_name} " + \
                                    f"{'has any value' if wildcard else f'= {value}'} by {reserved_by}"
                    st.session_state.chat_history.append({"role": "assistant", "content": success_message})
                    st.success(success_message)
                
                if already_reserved:
                    warning_message = f"⚠️ {len(already_reserved)} row(s) already reserved:\n" + \
                                   "\n".join([f"Row {idx}: reserved by {reserver}" for idx, reserver in already_reserved])
                    full_message = f"{warning_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
                    st.session_state.chat_history.append({"role": "assistant", "content": full_message})
                    st.warning(warning_message)
                    st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
                
                return reserved_count > 0
                
            except Exception as e:
                failure_message = f"❌ Error during reservation: {str(e)}"
                full_message = f"{failure_message}\nPlease [click here]({jira_link}) to raise a request in Jira."
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": full_message
                })
                st.error(failure_message)
                st.markdown(f"Please [click here]({jira_link}) to raise a request in Jira.")
                return False
        else:
            available_columns = ", ".join(st.session_state.df.columns) if st.session_state.df is not None else "No columns available"
            failure_message = f"❌ Invalid column name '{column_name}'. Available columns: {available_columns}"
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

    def run_chat_page(self):
        st.title("🤖 V-TRAIN")
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
            
            # Handle prompts irrespective of grammar, punctuation, or question marks
            prompt_lower = user_prompt.lower().replace("?", "").replace(",", "").replace(".", "").strip()
            prompt_words = set(prompt_lower.split())
            # Handle German prompts irrespective of grammar, punctuation, or question marks
            prompt_lower = user_prompt.lower().replace("?", "").replace(",", "").replace(".", "").strip()
            prompt_words = set(prompt_lower.split())
            
            # Prompt 1: "Hallo wie geht es dir?"
            hello_words = {"hallo", "wie", "geht", "es", "dir"}
            if hello_words.issubset(prompt_words):
                response = "Mir geht es gut. Wie kann ich Ihnen heute behilflich sein?"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            # Prompt 2: "Ich brauche Ihre Hilfe, um die Daten für meinen Test zu finden"
            elif {"ich", "brauche", "hilfe", "daten", "test", "finden"}.issubset(prompt_words):
                response = "Ja, sicher. Können Sie mir mit den Einzelheiten helfen?"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

            # Prompt 3: "Ich brauche zwei Kunden-IDs, die migriert werden"
            elif {"ich", "brauche", "zwei", "kunden-ids", "migriert"}.issubset(prompt_words):
                response = "Hier sind zwei migrierte Kunden - 1009010, 1009014."
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            # Prompt 4: "Ich brauche zwei Kunden-IDs, die migriert werden"
            elif {"ich", "brauche", "zwei", "kunden-ids", "migriert"}.issubset(prompt_words):
                migrated_ids = self.get_migrated_customer_ids(2)
                if len(migrated_ids) == 2:
                    response = f"Hier sind zwei migrierte Kunden - {migrated_ids[0]}, {migrated_ids[1]}."
                elif migrated_ids:
                    response = f"Nur ein migrierter Kunde gefunden - {migrated_ids[0]}."
                else:
                    response = "Keine migrierten Kunden-IDs gefunden."
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

            # Prompt 1: "Hello"
            elif "hello" in prompt_words:
                response = "Hi, how may I assist you today?"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            # Prompt 2: "Can you provide me a data for my test"
            elif {"can","you", "provide", "me", "a", "data", "for", "my", "test"}.issubset(prompt_words):
                response = "Yes sure, could you please help me with more details?"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            # Prompt 3: "I need one mint registered customer id which is migrated"
            elif {"i", "need", "one", "mint", "registered", "customer", "id", "which", "is", "migrated"}.issubset(prompt_words):
                response = f"Please find the customer id which is mint registered and migrated- 10090098"
                reserve_message=f"✔️Successful, Data has been reserved for Abhijeet Waghmode"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": reserve_message})
                st.success(reserve_message)
            
            # Prompt 4: "Can you please reserve the data for me?"
            elif {"can", "you", "reserve", "data", "me"}.issubset(prompt_words):
                response = "✔️Successful, Data has been reserved for Abhijeet Waghmode"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                #with st.chat_message("assistant"):
                st.success(response)
            
            # Prompt 5: "Can you show me the serve data?" (assuming "serve" might be a typo for "reserved")
            elif {"can", "you", "show", "me", "data"}.issubset(prompt_words) and ("serve" in prompt_words or "reserved" in prompt_words):
                response = "Please find the data which is reserved by Abhijeet Waghmode - 10090098"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)


            #Mint data
            # Prompt 1: "Hello, how are you??"
            elif {"Hello", "how", "are", "you"}.issubset(prompt_words):
                response = "Hello Abhijeet! I'm here and ready to help you with anything you need. How can I assist you today? 😊"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            # Prompt 2: "I need your help in finding mint credentials for two fusionC customers."
            elif {"I","need", "your", "help", "in", "finding", "mint", "credentials", "for", "two", "fusionC", "customers"}.issubset(prompt_words):
                response = "Sure! I can help you with that. Please provide the customer IDs so I can fetch the mint credentials for them."
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            # Prompt 3: "The customer Ids are- 10090012 and 10090044."
            elif {"The", "customer", "Ids", "are", "10090012", "and", "10090044"}.issubset(prompt_words):
                response = f"I have found the mint credentials for the requested customers:" +\
                           f"Customer Id 10090044: Username – test_10090012@vodafone.com, Password – Test@111"
                reserve_message=f"✔️Successful, 10090044 has been reserved for Abhijeet Waghmode"
                error_message=f"⚠️Customer Id 10090012: This data is currently reserved for user Nilotpal Das. Please contact him for the mint credentials."
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": reserve_message})
                st.success(reserve_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                st.error(error_message)
            
            # Prompt 4: "Can you please reserve the data for me?"
            elif {"can", "you", "reserve", "data", "me"}.issubset(prompt_words):
                response = "✔️Successful, Data has been reserved for Abhijeet Waghmode"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                #with st.chat_message("assistant"):
                st.success(response)
            
            # Existing reservation logic
            else:
                row_number = self.extract_row_number(user_prompt)
                column_name, column_value = self.extract_column_value_reservation(user_prompt)
                with_column_name, with_value = self.extract_with_pattern_reservation(user_prompt)
                
                if row_number is not None:
                    self.reserve_data(row_number)
                    #st.write("Updated data:")
                    #st.write(st.session_state.df)
                elif column_name is not None and column_value is not None:
                    self.reserve_by_column_value(column_name, column_value)
                    #st.write("Updated data:")
                    #st.write(st.session_state.df)
                elif with_column_name is not None and with_value is not None:
                    is_wildcard = with_value.strip() == "*"
                    self.reserve_by_column_value(with_column_name, with_value, wildcard=is_wildcard)
                    #st.write("Updated data:")
                    #st.write(st.session_state.df)
                elif "reserve" in user_prompt.lower():
                    response = "Please specify either:\n1. A row number (e.g., 'reserve row 5')\n" + \
                              "2. A column and value (e.g., 'reserve data where column_name = value')\n" + \
                              "3. Data with pattern (e.g., 'reserve the data with customer - 100900249502' or 'reserve the data with customer - *')"
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

def login_page():
    st.title("Login to V-TRAIN")
    st.write("Enter your credentials below:")
    
    with st.form(key='login_form'):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login")
        
        if submit_button:
            # Allow login with or without credentials (no validation)
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()  # Rerun to redirect to chat page

def main():
    app = DataFrameChat()
    
    if not st.session_state.logged_in:
        login_page()
    else:
        app.run_chat_page()

if __name__ == "__main__":
    main()
