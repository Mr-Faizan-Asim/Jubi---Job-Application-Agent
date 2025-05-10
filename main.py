from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import re

# Load your environment variables from .env file
load_dotenv()

# Connect to Groq using OpenAI-compatible interface with LLaMA model
llm = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_API_BASE"),  # Example: https://api.groq.com/openai/v1
    model="llama2-70b-4096",              # ✅ LLaMA model available on Groq
    temperature=0
)

# Memory to hold conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Data store for application information
application_info = {
    "name": None,
    "email": None,
    "skills": None
}

# Tool 1: Extract application details
def extract_application_info(text: str) -> str:
    name_match = re.search(r"(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.IGNORECASE)
    email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
    skills_match = re.search(r"(?:skills are|i know|i can use)\s+(.+)", text, re.IGNORECASE)

    response = []

    if name_match:
        application_info["name"] = name_match.group(1).title()
        response.append("✅ Name saved.")

    if email_match:
        application_info["email"] = email_match.group(0)
        response.append("✅ Email saved.")

    if skills_match:
        application_info["skills"] = skills_match.group(1).strip()
        response.append("✅ Skills saved.")

    if not any([name_match, email_match, skills_match]):
        return "❓ I couldn't extract any info. Could you please provide your name, email, or skills?"

    return " ".join(response) + " Let me check what else I need."

# Tool 2: Check if all data is collected
def check_application_goal(_: str) -> str:
    if all(application_info.values()):
        return f"✅ You're ready! Name: {application_info['name']}, Email: {application_info['email']}, Skills: {application_info['skills']}."
    else:
        missing = [k for k, v in application_info.items() if not v]
        return f"⏳ Still need: {', '.join(missing)}. Please ask the user to provide this."

# Define the tools for the agent
tools = [
    Tool(
        name="extract_application_info",
        func=extract_application_info,
        description="Extracts name, email, and skills."
    ),
    Tool(
        name="check_application_goal",
        func=check_application_goal,
        description="Checks if name, email, and skills are provided.",
        return_direct=True
    )
]

# Define system prompt to guide the agent behavior
SYSTEM_PROMPT = """You are a helpful job application assistant. 
Your goal is to collect the user's name, email, and skills. 
Use the tools provided to extract this information and check whether all required data is collected.
Once everything is collected, inform the user that the application info is complete and stop.
"""

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": SYSTEM_PROMPT}
)

# Start interaction
print("📝 Hi! I'm your job application assistant. Please tell me your name, email, and skills.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Bye! Good luck.")
        break

    response = agent.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Stop if all data is collected
    if "you're ready" in response["output"].lower():
        print("🎉 Application info complete!")
        break
