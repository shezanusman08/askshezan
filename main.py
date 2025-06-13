import streamlit as st
import os
from dotenv import load_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner, RunConfig

# Load environment variables
load_dotenv()

# Initialize the AI model and configuration
MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model=MODEL_NAME,
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define assistants
assistants = {
    "Manager": Agent(
        name="Manager at Software House",
        instructions="Your job is to respond to the user's query in a professional and managerial manner. Say no to any requests that are not related to software development, web development, or digital marketing.",
        model=model
    ),
    "Website Developer": Agent(
        name="Website Developer",
        instructions="You are a professional website developer. Just take orders and respond technically about website development. say no to any requests that are not related to website development.",
        model=model
    ),
    "Digital Marketer / SEO Expert": Agent(
        name="Digital Marketing Expert",
        instructions="You are a digital marketing expert. Help users with SEO strategies, keywords, and business promotion ideas. say no to any requests that are not related to digital marketing.",
        model=model
    )
}

# Streamlit UI
st.set_page_config(page_title="Software House Agent", page_icon="💼")
st.title("💻 AskShezan AI")
st.write("Tech, Web, and Marketing — all in one chat.")
st.write("Choose a department to interact with:")


# User input
selected_assistant = st.selectbox("Choose an assistant:", list(assistants.keys()))
user_input = st.text_area("Write your message:", placeholder="e.g. I want a website for my bakery business")

if st.button("Submit"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Thinking..."):
            assistant = assistants[selected_assistant]
            import asyncio
            result = asyncio.run(Runner.run(assistant, user_input, run_config=config))
            st.success("Response:")
            st.write(result.final_output)
