from crewai import Agent, Crew, Task
from crewai_tools import MCPServerAdapter
from llama_cpp import Llama
from mcp import StdioServerParameters
import os

# Load Kimi-K2 GGUF via llama.cpp
llm = Llama(
    model_path=r".\models\Kimi-K2-Instruct-Q2_K-00001-of-00008.gguf", 
    n_ctx=4096,
    n_gpu_layers=-1,
    n_threads=8,
    )

# Connect to your FastMCP Chrome server
#chrome_adapter = MCPServerAdapter(server_url="http://localhost:9000")
# Use stdio instead of server_url
server_params = StdioServerParameters(
    command="uv",
    args=["run", "python", "server.py"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with MCPServerAdapter(server_params) as tools:
    #chrome_adapter = MCPServerAdapter(
    #    command=["uv", "run", "python", "server.py"],  # spawns your FastMCP server
    #)

    print(f"Available tools from Stdio MCP server: {[tool.name for tool in tools]}")

    browser_agent = Agent(
        role="Web Navigator",
        goal="Open pages and summarize content",
        backstory="Autonomous agent using Chrome browsing tools",
        llm=llm,
        #adapters=[chrome_adapter],  # attach MCP server adapter
        tools=tools,
        verbose=True,
    )

    task = Task(
        description="Open https://example.com, extract structured content, and summarize it.",
        agent=browser_agent
    )

    crew = Crew(
        agents=[browser_agent], 
        tasks=[task],
        verbose=True,
        )

    result = crew.kickoff(inputs={"website": "open_url(https://www.heise.de/)"})
    print(result)

#if __name__ == "__main__":
#    crew.kickoff()
