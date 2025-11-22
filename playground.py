
from phi.agent import Agent

from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
load_dotenv()

import os
import phi
from phi.playground import Playground, serve_playground_app

phi.api = os.getenv("PHI_API_KEY")

web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for information",
    tools=[DuckDuckGo()],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)


finance_agent=Agent(
    name="Finance Agent",
    role="Answer questions about finance",
    tools=[
        YFinanceTools(stock_fundamentals=True, stock_price=True,analyst_recommendations=True,company_news=True)],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["use tables to display the information"],
    show_tools_calls=True,
    markdown=True,
)


app =Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)