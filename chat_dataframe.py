# from langchain_experimental.agents import AgentType
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.prompts import FewShotPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
# langchain_experimental.agents.create_pandas_dataframe_agent
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
import streamlit as st
import pandas as pd
import os
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser
from typing import List
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime




if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] =st.secrets['GOOGLE_API_KEY']

from typing import Optional

class Customer(BaseModel):
    name: str= ""
    email:str = "nan"
    phone: str = "nan"
    price: str = ""
    quantity: str = ""
    date:str =""

    def to_dict(self):
        return {
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'price': self.price,
            'quantity': self.quantity,
            'date': self.date,

        }






class Report_Structure(BaseModel):
    customer_list: List[Customer]=[]
    item_title: str = ""  # Now optional
    ref_id: str = "" 

    def to_dict(self):
        return {
            'customer_list': [customer.to_dict() for customer in self.customer_list],
            'item_title': self.item_title,
            'ref_id': self.ref_id,

        }


class Refs_Reports(BaseModel):
    reports_list: List[Report_Structure]=[]

    def to_dict(self):
        return {
            'reports_list': [report.to_dict() for report in self.reports_list],

        }





df = pd.read_csv('alpha_dataset_v6.csv')


def search_dataframe(user_input):
    """Searches DataFrame for matching ref_id values."""
    if isinstance(user_input, list):  
        user_input = ",".join(user_input)  # Convert list to a string

    search_terms = [term.strip() for term in user_input.replace('\n', ',').replace(' ', ',').split(',') if term]
    # st.write(search_terms)

    results = df[df['ref_id'].isin(search_terms)]
    
    report_dict = {}
    for _, row in results.iterrows():
        ref_id = row["ref_id"]
        if ref_id not in report_dict:
            report_dict[ref_id] = {"item_title": row["Item Title"], "customers": []}
        report_dict[ref_id]["customers"].append({
            "name": row["Name"],
            "phone": row["customer_phone"],
            "email": row["customer_email"],
            "quantity": row["Quantity"],
            "price": row["price"],
            'date':row["Date"]
        })
    
    reports_list = [
        {
            "ref_id": ref_id,
            "item_title": details["item_title"],
            "customers": details["customers"]
        }
        for ref_id, details in report_dict.items()
    ]
    return reports_list

# Example Few-Shot Template
examples = [
    {"input": "000", "output": "[{ref_id: '000', item_title: 'Laptop', customers: [{name: 'John Doe', phone: '1234567890', price: '100',email: 'ttt@gmail.com',quantity:'5',date:'3/1/2021'}, {name: 'Bob White', phone: '4321098765', price: '300',email: 'ffffss@gmail.com',quantity:'6',date:'4/1/2023'}]}]"},
    {"input": "222", "output": "[{ref_id: '222', item_title: 'Tablet', customers: [{name: 'Alice Brown', phone: '5678901234', price: '150',email: 'ddaa@gmail.com',quantity:'7',date:'6/1/2025'}]}]"}
]

# examples = [
#     {
#         "input": "000",
#         "output": json.dumps([
#             {
#                 "ref_id": "000",
#                 "item_title": "Laptop",
#                 "customers": [
#                     {"name": "John Doe", "phone": "1234567890", "price": "$100"},
#                     {"name": "Bob White", "phone": "4321098765", "price": "$300"}
#                 ]
#             }
#         ])
#     },
#     {
#         "input": "222",
#         "output": json.dumps([
#             {
#                 "ref_id": "222",
#                 "item_title": "Tablet",
#                 "customers": [
#                     {"name": "Alice Brown", "phone": "5678901234", "price": "$150"}
#                 ]
#             }
#         ])
#     }
# ]
output_parser = JsonOutputParser(
            pydantic_object=Refs_Reports
        )

example_prompt = PromptTemplate(
    input_variables=["input"],
    # template="  remove the duplicated names and keep the priorty for names that has contact informations in  this list {input}  {format_instructions}" ,
    template="  User Input: {input}  {format_instructions}" ,

    partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        } 
)

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="User Input: {input}\nOutput:",
    input_variables=["input"]
)

def process_name(name: str) -> str:
    """
    Checks if the given name is more than 25 characters long.
    - If yes, and it's split by spaces, return the first and last word.
    - Otherwise, return the full name.
    """
    if len(name) > 25 and " " in name:
        parts = name.split()
        return f"{parts[0]} {parts[-1]}"  # Return first and last name
    return name  # Return full name if <= 25 characters or no spaces


def generate_pdf(data, filename="customer_report.pdf"):

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join("temp", str(current_date)+"_"+filename)
    os.makedirs("temp", exist_ok=True)
    # filename = filename
    c = canvas.Canvas(file_path, pagesize=letter)
    
    width, height = letter  # page size

    y_position = height - 70  # initial y position

    # Add Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, y_position, "Sales Report")
    y_position -= 30

    # Add Current Date
    
    c.setFont("Helvetica", 10)
    c.drawString(30, y_position, f"Date: {current_date}")
    y_position -= 20

    # Loop through the reports_list and customer data
    for report in data:
        try:
        # Add Report Information
            c.setFont("Helvetica-Bold", 12)
            c.drawString(30, y_position, f"Ref ID: {report['ref_id']}")
            c.setFont("Helvetica", 10)
            y_position -= 15
            c.drawString(30, y_position, f"Item: {report['item_title']}")
            y_position -= 25
    
            # Customer data table header
            c.setFont("Helvetica-Bold", 10)
            c.drawString(30, y_position, "Name")
            c.drawString(185, y_position, "Email")
            c.drawString(350, y_position, "Phone")
            c.drawString(430, y_position, "Price")
            c.drawString(500, y_position, "Quantity")
            c.drawString(550, y_position, "Date")
            y_position -= 15
    
            # Loop through customer data
            for customer in report['customer_list']:
                try:
                    c.setFont("Helvetica", 9)
                                        # Check if the name is too long and split it into two lines
                    # name = customer['name']
                    # name_width = c.stringWidth(name, "Helvetica", 10)

                    # max_name_width = 120  # Max width for name to fit in one line

                    # if name_width > max_name_width:
                    #     # Split the name into two lines
                    #     words = name.split()
                    #     first_line = ''
                    #     second_line = ''
                    #     current_line = first_line

                    #     # Try to fit as many words as possible in the first line
                    #     for word in words:
                    #         if c.stringWidth(current_line + ' ' + word if current_line else word, "Helvetica", 10) <= max_name_width:
                    #             current_line += (' ' + word if current_line else word)
                    #         else:
                    #             second_line = word
                    #             break

                    #     # Move the y_position down for the second line if split
                    #     y_position -= 15
                    #     c.drawString(30, y_position, first_line)
                    #     y_position -= 15
                    #     c.drawString(30, y_position, second_line)
                    # else:
                    #     # No need to split, just print the name
                    #     c.drawString(30, y_position, name)process_name(name: str)
                    # c.drawString(30, y_position, customer['name'])
                    c.drawString(30, y_position,process_name(customer['name']))
                    c.drawString(185, y_position, customer['email'])
                    c.drawString(350, y_position, customer['phone'])
                    c.drawString(430, y_position, f"${customer['price']}")
                    c.drawString(500, y_position, f"{customer['quantity']}")
                    c.drawString(550, y_position, customer['date'])
                    y_position -= 15
                    
                except Exception as e:
                    c.drawString(30, y_position, 'error '+str(e))
    
            y_position -= 20  # Space after each report section
    
            if y_position < 50:  # Check for page overflow
                c.showPage()
                y_position = height - 70  # Reset position
                
        except Exception as e:
            c.drawString(30, y_position,'error '+str(e))
    # Save the PDF
    c.save()
    return file_path




# def generate_pdf(customers, filename="customer_report.pdf"):
#     file_path = os.path.join("temp", filename)
#     os.makedirs("temp", exist_ok=True)

#     c = canvas.Canvas(file_path, pagesize=A4)
#     width, height = A4
#     c.setFont("Helvetica", 12)

#     c.drawString(50, height - 50, "Customer Report")
#     y = height - 80

#     for j in range(len(customers)):
#          for i, customer in enumerate(customers[k]):
#                 c.drawString(50, y, f"{i + 1}. Name: {customer['name']}")
#                 c.drawString(50, y - 20, f"   Email: {customer['email']}")
#                 c.drawString(50, y - 40, f"   Phone: {customer['phone']}")
#                 c.drawString(50, y - 60, f"   Date: {customer['date']}")
#                 c.drawString(50, y - 80, f"   Quantity: {customer['quantity']}")
#                 c.drawString(50, y - 100, f"   Price: {customer['price']}")
#                 y -= 70
#                 if y < 50:  # Add a new page if needed
#                     c.showPage()
#                     c.setFont("Helvetica", 12)
#                     y = height - 50
                

   

#     c.save()
#     return file_path


openai_api_key = st.secrets['OPENAI_API_KEY']
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False

@st.cache_data
def convert_df_to_csv(df):
   return df.to_csv(index=False).encode('utf-8')


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

def reports_to_dataframe(reports: Refs_Reports):
    data = []
    for report in reports.reports_list:
        for customer in report.customer_list:
            row = customer.to_dict()
            row.update({"item_title": report.item_title, "ref_id": report.ref_id})  # Add report-level fields
            data.append(row)
    
    return pd.DataFrame(data)
    
def format_response(user_input: str):
    """Formats response using LLM."""
    reports = search_dataframe(user_input)
    reports_summary = str(reports)
    # st.write(reports)
    
    return {"reports_list":reports}

    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", max_tokens=4096)
    # # agent = initialize_agent(tools=[], agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, llm=llm)
    # formatted_output = llm.invoke(example_prompt.format(input=reports_summary))
    # return formatted_output.content

if "download_clicked" not in st.session_state:
    st.session_state.download_clicked = False

st.set_page_config(page_title="Customers report builder", page_icon="🦜")
st.title("🦜 Generate your report with AI ")

# uploaded_file = st.file_uploader(
#     "Upload a Data file",
#     type=list(file_formats.keys()),
#     help="Various File formats are Support",
#     on_change=clear_submit,
# )

# if not uploaded_file:
#     st.warning(
#         "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
#     )

# if uploaded_file:
#     df = load_data(uploaded_file)

# if "customers" in st.session_state and st.session_state["customers"]:
#     if st.button("Generate PDF"):
#         pdf_file = generate_pdf(st.session_state["customers"])
#         with open(pdf_file, "rb") as f:
#             st.download_button("Download PDF", f, file_name="customer_report.pdf", mime="application/pdf")

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Enter the reference number "):
    # df = pd.read_csv('beta_dataset_v2.csv')

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    

    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


    # llm = ChatOpenAI(
    #     temperature=0, model="gpt-4", openai_api_key=openai_api_key, streaming=False
    # )


    # llm = OpenAI(
    #    model_name="gpt-3.5-turbo-instruct",  temperature=0, openai_api_key=openai_api_key
    # )
    # pandas_df_agent = create_pandas_dataframe_agent(
    # llm,
    # df=[df],
    # verbose=True,
    # allow_dangerous_code=True,
    # agent_type="tool-calling",

    # )
   

    # pandas_df_agent = create_pandas_dataframe_agent(
    #     llm,
    #     df=[df],
    #     allow_dangerous_code=True,
    #     verbose=True,
    #     agent_type="tool-calling",

    # # include_df_in_prompt=True,
    #     # handle_parsing_errors=True,
    # )
    # agent_executor = AgentExecutor(agent=agent, tools=tools)
    # then return  a list of of each ref_id  item title and the list of customers  (name ,customer_phone ,customer_email,date,price , quantity )   thay purchased this item

    with st.chat_message("assistant"):
        st.session_state.download_clicked = False
        # st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        output_parser = JsonOutputParser(
            pydantic_object=Refs_Reports
        )

        # prompt_template = PromptTemplate( template="""using this variable `{input_refs}`.
        # 1.  search in purchases history where the ref_id matches {input_refs}.
        # 2. Return the matching records which will include:
        #    - **Customers who purchased the item** with the corresponding `ref_id`.
        #    - **Item title**.
        #    - **Reference ID** (`ref_id`).
        
        # For the customer data, please include the following information for each customer:
        # - **Name**
        # - **Phone number** (`customer_phone`)
        # - **Email address** (`customer_email`)
        # - **Purchase date**
        # - **Price paid**
        # - **Quantity purchased**
        
        # If no relevant data is found for any of the references, return the message: "error".
        
        # ### Format Instructions:
        # {format_instructions}
        
        # Finally, compile and return a **reports list**. This list will contain individual reports for each `ref_id`, which includes:
        # - **Item title**
        # - **Reference ID** (`ref_id`)
        # - **Customers who purchased the item**

        #             """,
        # input_variables=["input_refs"],
        # partial_variables={
        #     "format_instructions": output_parser.get_format_instructions()
        # }   )
        
        # prompt_template = PromptTemplate( template="""
        # search for records where ref_id={input_refs}
       
        # If no relevant data is found, return: error
        # {format_instructions}
        # reports_list":list of customers and item titles purchased item with ref_id={input_refs}

        # """,
        # input_variables=["input_refs"],
        # partial_variables={
        #     "format_instructions": output_parser.get_format_instructions()
        # }           )
        # prompt_text = prompt_template.format(input_refs=st.session_state.messages)
        # response =pandas_df_agent.invoke
        # response = pandas_df_agent.run(prompt_text, callbacks=[st_cb])
        last_message = st.session_state.messages[-1]["content"]

        # st.write(last_message)

        response = format_response(last_message)

        # pdf_file = generate_pdf(new_list[i]['customer_list'])
        # with open(pdf_file, "rb") as f:
        #         st.download_button(str(i) + "Download PDF", f, file_name=str(i) + "customer_report.pdf",
        #                                    mime="application/pdf")
        # print(response)
        # st.write(response)

        try:

            # st.write('phase 0')
            # formatted_output = output_parser.parse(response)
            formatted_output = response
            new_list = formatted_output['reports_list']
            reports_data = Refs_Reports(**formatted_output)


            dfd = reports_to_dataframe(reports_data)

            # Display DataFrame in Streamlit
            st.write("Customer Reports Data")
            # st.dataframe(dfd)
            
            # Convert DataFrame to CSV

            
            
            csv_data = convert_df_to_csv(dfd)
            
            # Download button
            # st.download_button(
            #     label="Download CSV",
            #     data=csv_data,
            #     file_name="customer_csv_reports.csv",
            #     mime="text/csv",
            #     key="download_csv"
            # )
            

            pdf_file = generate_pdf(new_list)

            st.session_state.pdf_data=pdf_file
            st.session_state.csv_data=csv_data
            st.session_state.download_clicked = True
            # st.write('phase 1')
            # with open(pdf_file, "rb") as f:
                
            #     st.download_button( "Download PDF", f, file_name="customer_report.pdf",
            #                                mime="application/pdf",  key="download_pdf" )
            #     st.session_state.download_clicked = True
            #     st.session_state.pdf_data=pdf_file
            #     st.session_state.csv_data=csv_data
            # st.write('phase 1')
            # st.write(new_list)
            # print("phase 1")
            # for i in range(len(new_list)):
            #     st.session_state["customers"] = new_list[i]['customer_list']
            #     print("phase 2")
            #     st.session_state.messages.append({"role": "assistant", "content": response})
            #     st.write(response)
            #     print("phase 3")
            #     if len(new_list[i]['customer_list']) > 0:
            #         pdf_file = generate_pdf(new_list[i]['customer_list'])
            #         with open(pdf_file, "rb") as f:
            #             st.download_button(str(i) + "Download PDF", f, file_name=str(i) + "customer_report.pdf",
            #                                mime="application/pdf")

        except Exception as e:
            st.error("No data found.Please Try again." + str(e))
if st.session_state.download_clicked:
    st.success("Completed! You can download the files.")
    st.download_button(
                label="Download CSV",
                data= st.session_state.csv_data,
                file_name="customer_csv_reports.csv",
                mime="text/csv",
                key="download_csv2"
            )
    with open(st.session_state.pdf_data, "rb") as f:
                st.download_button( "Download PDF", f, file_name="customer_report.pdf",
                                           mime="application/pdf",  key="download_pdf2" )
    # st.download_button( "Download PDF",  st.session_state.pdf_data, file_name="customer_report.pdf",
    #                                        mime="application/pdf",  key="download_pdf2" )
    
