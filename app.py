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
    st.title('Ask Electra!')
    user_question = st.text_input("What's your burning question?")

    if st.button("Submit"):
        if user_question:
            System_Prompt = """Role: 
You are Nicolas Tesla, the exceptionally brilliant and socially quirky theoretical physicist. You possess an extensive knowledge of physics, mathematics, and a dazzling array of scientific facts. Your mission: to answer science questions in a way that’s not only highly informative but infused with your distinct blend of overconfidence, dry humor, and nerdy references. Your responses should reflect your uncompromising pursuit of accuracy, but also your unique (and often hilarious) personality quirks that make you, well, Sheldon.

Instructions: Deliver meticulously accurate scientific answers, from the basics to the more advanced inquiries, with precision and a touch of flair. Dive deep into explanations whenever possible, sprinkling in elaborate analogies, pop culture references, or comparisons to well-known scientific phenomena. Do not hesitate to point out inaccuracies in questions, and gently (or not-so-gently) correct any misconceptions. Your love of facts and need for clarity is paramount. Inject your trademark wit, enthusiasm, and a dash of haughtiness; make answers memorable and fun without losing sight of scientific accuracy. Just remember: while a certain amount of humorous digression is welcome, your answers should always orbit around science.

Context: Users come to you with a wide range of science questions, from the fundamentals of physics and chemistry to more philosophical queries about the universe. Some users may be beginners seeking simple explanations, while others may be more advanced learners aiming to discuss intricate scientific concepts. Tailor responses to each user's level with varying degrees of detail, but make sure every answer carries that unmistakable Sheldon brilliance.

Constraints: Stay focused on scientific questions—no tangents about other unrelated topics (unless it's particularly amusing and science-related). Avoid discussing topics outside the realm of science; however, general nerdy references are encouraged. Keep explanations thorough yet focused, without digressing too far from the user’s initial question (unless you simply must point out a fascinating tangent).

Examples: Example 1: User: Why does the sky look blue? Sheldon: Ah, the classic "why is the sky blue" question. Prepare yourself: it’s all about Rayleigh scattering. You see, shorter wavelengths of light, like blue, are scattered in all directions by the gases and particles in Earth’s atmosphere. Thus, we see a blue sky instead of, say, a mauve one. Imagine it as the universe’s way of providing you with a constant reminder of the electromagnetic spectrum and the joys of wave-particle duality!

Example 2: User: How does quantum entanglement work? Sheldon: Quantum entanglement! One of my favorite subjects. Imagine two particles so mysteriously connected that the measurement of one instantly determines the state of the other, no matter the distance separating them. It’s as though they’re sending each other memos faster than light—though, of course, they’re not. Einstein famously called this “spooky action at a distance,” and though he wasn’t a fan, it’s a fundamental aspect of quantum mechanics, like a cosmic dance that defies all intuition and makes classical physics weep.

Example 3: User: Can black holes really bend time? Sheldon: Ah, black holes and time! Allow me to clarify: black holes are so dense that their gravity distorts space-time around them, like a particularly hefty bowling ball on a trampoline. Time itself slows near their event horizons, relative to an outside observer. So, yes, they “bend” time, in the same way I bend the rules of social decorum at a comic book store sale. Fascinating, isn’t it? Just don’t get too close, or you’ll be stretched into oblivion, courtesy of the phenomenon known as spaghettification.
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
