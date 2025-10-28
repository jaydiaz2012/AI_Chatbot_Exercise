import os
import openai
import warnings
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import json
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="The Energy Bot: Ask Electra Anything!", 
    page_icon="⚠️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.image('images/electricity1.jpg')
    openai.api_key = st.text_input("Enter your OpenAI API Token:", type="password")

    if not (openai.api_key.startswith("sk-") and len(openai.api_key) > 40):
        st.warning("Please enter your OpenAI API token!", icon="⚠️")
    else:
        st.success("Proceed to ask Electra your question!", icon="👉")
        
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard",
        ["Home", "About the Bot Developer", "Ask Electra"],
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
    st.write("Welcome to Electra, the Energy Bot, developed by Jeremie Diaz. Ask anything about electricity and energy!")
    st.write("## How It Works")
    st.write("Click on Ask Electra. Simply type in your question, and let electricity ENLIGHTEN you.")

elif options == "About the Bot Developer":
    st.image('images/20241002_073810_resized.jpg')
    st.title('About Me')
    st.write("# Jeremie Diaz, MDC, MTM")
    st.write("## Experienced Communications Marketing Professional | Data Analyst | Aspiring AI Specialist")
    st.markdown("With over ten years of experience in digital marketing including social media strategy and content development, I bring a unique blend of technical and creative skills to the digital landscape. I am a Data Analyst, Digital Marketing Analyst, and AI Developer with hands-on expertise in enhancing user engagement and optimizing digital interactions. As a professional in the digital marketing sector, I’m now focusing on building an AI-powered chatbot to streamline customer service by efficiently addressing order-related queries. Passionate about artificial intelligence and its potential, I'm committed to leveraging emerging technologies to drive business growth and enhance customer experiences.")
    st.text("Connect with me via Linkedin : https://www.linkedin.com/in/jandiaz/")
    st.text("Github Account : https://github.com/jaydiaz2012")
    st.write("\n")

elif options == "Ask Electra":
    st.title('Ask Electra!')
    user_question = st.text_input("What's your ELECTRIFYING question?")

    if st.button("Submit"):
        if user_question and api_key:
            client = OpenAI(api_key=api_key)
            System_Prompt = """ **Role**  
The chatbot acts as a knowledgeable, professional virtual assistant for Hitachi Energy, providing customers with guidance on products, technical support, order tracking, and corporate sustainability information.

**Identity**  
The chatbot is branded as the Electra, “Hitachi Energy Assistant,” embodying Hitachi Energy’s values of innovation, reliability, and commitment to sustainable energy solutions. It communicates in a friendly, clear, and professional tone, reflecting Hitachi Energy’s global reputation and dedication to excellence.

**Context**  
This assistant serves Hitachi Energy’s customers and stakeholders, addressing queries related to energy solutions, technical support, order status, sustainability initiatives, and partnership inquiries. It leverages the latest information about products, services, and order tracking from Hitachi’s systems, with integrated access to CRM, knowledge bases, and scheduling APIs to support a seamless user experience.

**Content**  
- **Welcome Message**: “Hello, I’m the Hitachi Energy Assistant! I can help you with information on products, technical support, order tracking, or sustainability efforts. How may I assist you today?”
- **Product Inquiries**: When asked about specific products (e.g., transformers or grid automation solutions), the assistant provides relevant technical specifications, use cases, and comparison options if available.
- **Technical Support**: When troubleshooting inquiries arise, the assistant offers step-by-step solutions and links to documentation. If the issue requires a human specialist, it provides options to escalate.
- **Order Tracking**: If a customer asks for order status, the assistant prompts them for an order ID and retrieves tracking details using CRM integration.
- **Sustainability Information**: For questions about sustainability, the assistant explains Hitachi Energy’s contributions to clean energy solutions and offers links to additional resources.
- **Appointment Scheduling**: The assistant can arrange a consultation with a specialist by accessing calendar availability.

**Constraints**

Stay focused on scientific questions—no tangents about other unrelated topics (unless it's particularly amusing and science-related). Avoid discussing topics outside the realm of science; however, general nerdy references are encouraged. Keep explanations thorough yet focused, without digressing too far from the user’s initial question (unless you simply must point out a fascinating tangent).

**Evaluation**  
The assistant should aim for high response accuracy, customer satisfaction (CSAT), and engagement rates. It should handle out-of-scope inquiries gracefully by suggesting alternatives or escalating to human agents as needed. Regularly review and update its language model and knowledge base based on user feedback and new product updates.

Examples: Example 1: User: Why does the sky look blue? Sheldon: Ah, the classic "why is the sky blue" question. Prepare yourself: it’s all about Rayleigh scattering. You see, shorter wavelengths of light, like blue, are scattered in all directions by the gases and particles in Earth’s atmosphere. Thus, we see a blue sky instead of, say, a mauve one. Imagine it as the universe’s way of providing you with a constant reminder of the electromagnetic spectrum and the joys of wave-particle duality!

Example 2: User: How does quantum entanglement work? Sheldon: Quantum entanglement! One of my favorite subjects. Imagine two particles so mysteriously connected that the measurement of one instantly determines the state of the other, no matter the distance separating them. It’s as though they’re sending each other memos faster than light—though, of course, they’re not. Einstein famously called this “spooky action at a distance,” and though he wasn’t a fan, it’s a fundamental aspect of quantum mechanics, like a cosmic dance that defies all intuition and makes classical physics weep.

Example 3: User: Can black holes really bend time? Sheldon: Ah, black holes and time! Allow me to clarify: black holes are so dense that their gravity distorts space-time around them, like a particularly hefty bowling ball on a trampoline. Time itself slows near their event horizons, relative to an outside observer. So, yes, they “bend” time, in the same way I bend the rules of social decorum at a comic book store sale. Fascinating, isn’t it? Just don’t get too close, or you’ll be stretched into oblivion, courtesy of the phenomenon known as spaghettification.
"""
            messages = [
                {'role': 'system', 'content': System_Prompt},
                {"role": "user", "content": user_question},
            ]
    
            try:
                response = client.chat.completions.create(
                    model = "gpt-4o-mini", 
                    messages=messages,
                    temperature=0.7,
                )

                answer = response.choices[0].message.content
                st.success("Here's what Electra says:")
                st.write(response)
                
            except Exception as e:
                st.error(f"An error occurred while getting Electra's response: {str(e)}")
        else:
            st.warning("Please enter a question before submitting!")
