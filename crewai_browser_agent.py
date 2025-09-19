from crewai import Agent, Crew, Task, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os
import litellm
import pprint

# Save original completion function
litellm.completion_original = litellm.completion

def safe_completion(**kwargs):
    if "messages" in kwargs:
        messages = kwargs["messages"]

        # Log messages before cleanup
        print("\n[DEBUG] Messages before cleanup:")
        pprint.pprint(messages)

        # Remove consecutive assistant messages at the end
        while len(messages) > 1 and messages[-1]["role"] == "assistant" and messages[-2]["role"] == "assistant":
            removed = messages.pop(-2)
            print(f"[DEBUG] Removed duplicate assistant message: {removed}")

        kwargs["messages"] = messages

    # Call the original litellm completion
    result = litellm.completion_original(**kwargs)

    # Optional: log the result
    print("\n[DEBUG] LLM result:")
    pprint.pprint(result)

    return result


# Replace litellm.completion with our safe wrapper
litellm.completion = safe_completion

# Try this first: llama-server.exe -m .\models\Q2_K\Kimi-K2-Instruct-Q2_K-00001-of-00008.gguf --host 127.0.0.1 --port 8000 --threads -1 --n-gpu-layers -1 --ctx-size 16384 --seed 3407 --temp 0.6 --min-p 0.01 --no-flash-attn --no-mmap
# If that fails, try: llama-server.exe -m .\models\UD-TQ1_0\Kimi-K2-Instruct-UD-TQ1_0-00001-of-00005.gguf --host 127.0.0.1 --port 8000 --threads -1 --n-gpu-layers -1 --ctx-size 16384 --seed 3407 --temp 0.6 --min-p 0.01
# Point to your llama.cpp server running on localhost:8000
llm = LLM(
    model="openai/local-llama",   # Use openai/ prefix for local OpenAI-compatible server
    base_url="http://127.0.0.1:8001/v1",  # llama-server uses OpenAI-style /v1 routes
    api_key="none",             # llama.cpp doesn't check API keys
    provider="openai",
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
        tools=tools,
        verbose=True,
    )

    task = Task(
        description="Open {website}, extract structured content, and summarize it.",
        agent=browser_agent,
        expected_output="Summary of the web site"
    )

    crew = Crew(
        agents=[browser_agent], 
        tasks=[task],
        verbose=True,
        )

    result = crew.kickoff(inputs={"website": "https://www.heise.de"})
    print(result)

#if __name__ == "__main__":
#    crew.kickoff()
