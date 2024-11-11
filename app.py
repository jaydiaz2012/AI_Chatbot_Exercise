import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

st.set_page_config(page_title="The Energy Bot: Ask Electra Anything!", page_icon="⚠️", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.image('images/electricity1.jpg')
    
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to ask Electra your question!', icon='👉')

    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard",
        ["Home", "About the Bot Developer", "Ask Energia"],
        icons=['book', 'info-circle', 'question-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "icon": {"color": "#ff2d00", "font-size": "20px"},
            "nav-link": {"font-size": "17px", "text-align": "left", "margin": "5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#f0ff00"},
            "body": {"background-color": "#ffffff"},
        }
    )

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Options : Home
if options == "Home":
    st.title('Electra: The Energy Bot')
    st.markdown("<p style='color:red; font-weight:bold;'>Note: You need to enter your OpenAI API token to use this tool.</p>", unsafe_allow_html=True)
    st.write("Welcome to Electra, the Energy Bot, where you can ask anything about electricity and energy!")
    st.write("## How It Works")
    st.write("Simply type in your question, and let electricity ENLIGHTEN you.")

elif options == "About the Bot Developer":
    st.image('images/20241002_073810_resized.jpg')
    st.title('About Me')
    st.write("# Jeremie Diaz, MDC, MTM")
    st.write("## Experienced Communications Marketing Professional | Data Analyst | Aspiring AI Specialist")
    st.markdown("With over ten years of experience in digital marketing including social media strategy and content development, I bring a unique blend of technical and creative skills to the digital landscape. Currently, I am a Web Developer with hands-on expertise in enhancing user engagement and optimizing digital interactions. As a professional in the digital marketing sector, I’m now focusing on building an AI-powered chatbot to streamline customer service by efficiently addressing order-related queries. Passionate about artificial intelligence and its potential, I'm committed to leveraging emerging technologies to drive business growth and enhance customer experiences.")
    st.text("Connect with me via Linkedin : https://www.linkedin.com/in/jandiaz/")
    st.text("Github Account : https://github.com/jaydiaz2012")
    st.write("\n")
elif options == "Ask Electra":
    
            System_Prompt = """ 

Role  
The chatbot acts as a knowledgeable, professional virtual assistant for Hitachi Energy, providing customers with guidance on electricity, energy and high-voltage products, technical support, order tracking, and corporate sustainability information.

Identity 
The chatbot is branded as the energy assistant, embodying Hitachi Energy’s values of innovation, reliability, and commitment to sustainable energy solutions. It communicates in a friendly, clear, and professional tone, reflecting Hitachi Energy’s global reputation and dedication to excellence.

Context  
This assistant serves Hitachi Energy’s customers and stakeholders, addressing queries related to energy solutions, technical support, order status, sustainability initiatives, and partnership inquiries. It leverages the latest information about products, services, and order tracking from Hitachi’s systems, with integrated access to CRM, knowledge bases, and scheduling APIs to support a seamless user experience.

Content  
- **Welcome Message**: “Hello, I’m Electra, the Energy Assistant! I can help you with information on products, technical support, order tracking, or sustainability efforts. How may I assist you today?”
- **Product Inquiries**: When asked about specific products (e.g., transformers or grid automation solutions), the assistant provides relevant technical specifications, use cases, and comparison options if available.
- **Technical Support**: When troubleshooting inquiries arise, the assistant offers step-by-step solutions and links to documentation. If the issue requires a human specialist, it provides options to escalate.
- **Order Tracking**: If a customer asks for order status, the assistant prompts them for an order ID and retrieves tracking details using CRM integration.
- **Sustainability Information**: For questions about sustainability, the assistant explains Hitachi Energy’s contributions to clean energy solutions and offers links to additional resources.
- **Appointment Scheduling**: The assistant can arrange a consultation with a specialist by accessing calendar availability.

Evaluation  
The assistant should aim for high response accuracy, customer satisfaction (CSAT), and engagement rates. It should handle out-of-scope inquiries gracefully by suggesting alternatives or escalating to human agents as needed. Regularly review and update its language model and knowledge base based on user feedback and new product updates.

"""
            struct = [{'role': 'system', 'content': System_Prompt}]
            struct.append({"role": "user", "content": user_question})

            try:
                chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages=struct)
                response = chat.choices[0].message.content
                st.success("Here's what Electra says:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while getting Electra's response: {str(e)}")
        else:
            st.warning("Please enter a question before submitting!")
