import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
import streamlit as st

# langsmith tracing
os.environ['LANGHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']="Q&A Chatbot with Ollama Models"

# prompt template
prompt=ChatPromptTemplate.from_messages([
  ("system","You are a helpful assistant, respond to the user query"),
  ("user","{question}")
])

def generate_response(question,llm,temp,max_token):
  model=ChatOllama(model=llm,temperature=temp,num_predict=max_token)
  parser=StrOutputParser()
  chain=prompt|model|parser
  answer=chain.invoke({"question":question})
  return answer

# streamlit frontend
st.title("Q&A Chatbot with OLLAMA Models")
model_name=st.sidebar.selectbox("Select model:",["gemma:2b","llama2:latest","mxbai-embed-large:latest"])

# Adjust response param
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Token",min_value=50,max_value=300,value=150)

st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
  response=generate_response(user_input,model_name,temperature,max_tokens)
  st.write(response)
else:
  st.write("Please ask any question!")

