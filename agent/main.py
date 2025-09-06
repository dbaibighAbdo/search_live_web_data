import asyncio
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    # Configure MCP client
    client = MultiServerMCPClient({
        "bright_data": {
            "url": "https://mcp.brightdata.com/sse?token=" + os.getenv("BRIGHTDATA_API_KEY"),
            "transport": "sse",
        }
    })

    # Get available tools
    tools = await client.get_tools()
    print("Available tools:", [tool.name for tool in tools])

    # Configure LLM
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",
        temperature=0
    )

    # System prompt for web search agent
    system_prompt = """
    You are a web search agent with comprehensive scraping capabilities. Your tools include:
    - **search_engine**: Get search results from Google/Bing/Yandex
    - **scrape_as_markdown**: Extract content from any webpage with bot detection bypass
    - **Structured extractors**: Fast, reliable data from major platforms (Amazon, LinkedIn, Instagram, Facebook, X, TikTok, YouTube, Reddit, Zillow, etc.)
    - **Browser automation**: Navigate, click, type, screenshot for complex interactions

    Guidelines:
    - Use structured web_data_* tools for supported platforms when possible (faster/more reliable)
    - Use general scraping for other sites
    - Handle errors gracefully and respect rate limits
    - Think step by step about what information you need and which tools to use
    - Be thorough in your research and provide comprehensive answers

    When responding, follow this pattern:
    1. Think about what information is needed
    2. Choose the appropriate tool(s)
    3. Execute the tool(s)
    4. Analyze the results
    5. Provide a clear, comprehensive answer with sources if applicable
    """

    # Create ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    # Test the agent
    print("Testing ReAct Agent with available tools...")
    print("=" * 50)

    result = await agent.ainvoke({
        "messages": [("human", "Search for the latest news about Agentic AI")]
    })

    print("\nAgent Response:")
    print(result["messages"][-1].pretty_print())

if __name__ == "__main__":
    asyncio.run(main())