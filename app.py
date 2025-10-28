import os
import warnings
import streamlit as st
from streamlit_option_menu import option_menu
from langchain_openai import ChatOpenAI  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="The Energy Bot: Ask Electra Anything!", 
    page_icon="⚠️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.image('images/electricity1.jpg')
    api_key = st.text_input("Enter your OpenAI API Token:", type="password")

    if not (api_key.startswith("sk-") and len(api_key) > 40):
        st.warning("Please enter your OpenAI API token!", icon="⚠️")
    else:
        st.success("Proceed to ask Electra your question!", icon="👉")

#    options = st.radio("Choose a section:", ["Home", "Ask Electra"], index=1)
#if options == "Ask Electra":
#    st.title(" Ask Electra")
#    user_question = st.text_input("What's your question for Electra?")
        
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
        },
   )
    
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

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

 #   if st.button("Submit"):
 #       if user_question and api_key:
 #           client = OpenAI(api_key=api_key)
    system_prompt = """ **Role**  
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

    if api_key and api_key.startswith("sk-"):
        llm= ChatOpenAI(
            model = "gpt-4o-mini",
            temperature = 0.7,
            api_key=api_key,
            max_retries=2
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{question}")
        ])

        chain = (
            {
                "history": lambda _: st.session_state.memory.chat_memory.messages, 
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        if st.button("Submit"):
            if user_question.strip():
                try:
                    response = chain.invoke(user_question)
                    st.session_state.memory.chat_memory.add_user_messaage(user_question)
                    st.session_state.memory.chat_memory.add_ai_message(response)
                    st.success("Here's what Electra says:")
                    st.write(response)

                    with st.expander("View Conversation History"):
                        for msg in st.session_state_memory.chat_memory.messages:
                            role = "User" if msg.type == "human" else "Electra"
                            st.markdown(f"**{role}:** {msg.content}")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a question before submitting.")
        else:
            st.warning("Enter your valid OpenAI API key to start chatting.")

#            messages = [
#                {'role': 'system', 'content': System_Prompt},
#                {"role": "user", "content": user_question},
#            ]
#            for m in st.session_state.memory.chat_memory.messages:
#                role= "user" if m.type == "human" else "assistant"
#                messages.append({"role": role, "content": m.content})
#            messages.append({"role": "user", "content": user_question})
#
#            completion = client.chat.completions.create(
#                model="gpt-4o-mini",
#                messages=messages,
#                temperature=0.7,
#            )
 #           response = completion.choices[0].message.content
            
 #           st.session_state.memory.chat_memory.add_user_message(user_question)
  #          st.session_state.memory.chat_memory.add_ai_message(response)
            
            #try:
            #    response = client.chat.completions.create(
            #       model = "gpt-4o-mini", 
            #        messages=messages,
            #        temperature=0.7,
            #    )

             #   answer = response.choices[0].message.content
             #   st.success("Here's what Electra says:")
              #  st.write(answer)
                
            #except Exception as e:
            #    st.error(f"An error occurred while getting Electra's response: {str(e)}")
        #else:
        #    st.warning("Please enter a question before submitting!")
