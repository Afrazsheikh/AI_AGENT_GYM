# -------------------------------
# 🔹 Imports
# -------------------------------
import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType

# -------------------------------
# 🔹 Load ENV
# -------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------------
# 🔹 Tools
# -------------------------------
@tool
def calculate_nutrition(meal: str) -> str:
    """Calculate calories and protein"""
    
    meal = meal.lower()
    calories = 0
    protein = 0

    if "roti" in meal:
        calories += 120
        protein += 3

    if "egg" in meal:
        calories += 70
        protein += 6

    return f"Calories: {calories}, Protein: {protein}g"


@tool
def diet_advisor(data: str) -> str:
    """Give diet advice based on calories and protein"""
    return data


# -------------------------------
# 🔹 LLM (Groq API used here)
# -------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

# -------------------------------
# 🔹 Agent
# -------------------------------
agent = initialize_agent(
    tools=[calculate_nutrition, diet_advisor],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -------------------------------
# 🔹 Run
# -------------------------------
if __name__ == "__main__":
    query = input("Enter your meal: ")

    response = agent.run(query)

    print("\n✅ Output:\n")
    print(response)