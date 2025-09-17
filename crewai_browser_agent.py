from crewai import Agent, Crew, Task, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os

# Wrap llm using LiteLLM via CrewAI's LLM wrapper (expects an OpenAI-compatible endpoint)
# Example: run a local llama.cpp server
# First try via python - FAILED
#   python -m llama_cpp.server --model .\models\Kimi-K2-Instruct-Q2_K-00001-of-00008.gguf --host 127.0.0.1 --port 8000 --n_gpu_layers -1 --n_threads 8 --n_ctx 4096

# Second try
# llama-server.exe -m .\models\Q2_K\Kimi-K2-Instruct-Q2_K-00001-of-00008.gguf --host 127.0.0.1 --port 8000 --threads 16 --n-gpu-layers 30 --ctx-size 16384 --seed 3407 --temp 0.6 --min-p 0.01 --cache-type-k q4_0 -ot ".ffn_.*_exps.=CPU"
llm = LLM(
    model=os.getenv("LLM_MODEL", "llama-cpp"),
    base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.getenv("LLM_API_KEY", "NA"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
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
