from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo


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


multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Always include sources","Use table to display the data"],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("What is the stock price of Apple and share the latest news regarding it", stream=True)