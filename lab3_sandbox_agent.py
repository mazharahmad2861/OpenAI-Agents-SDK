import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Runner, Agent
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.sandbox import (
    Manifest,
    SandboxAgent,
    SandboxRunConfig,
    SandboxPathGrant,
)
from agents.sandbox.capabilities import Capabilities
from agents.sandbox.entries import LocalDir
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# 2. Groq client
# ============================================================

groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=groq_client,
)


# ============================================================
# 3. Folders
# ============================================================

CODE_DIR = Path("code").resolve()
OUTPUT_DIR = Path("output").resolve()

CODE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Create a sample Python file
sample_file = CODE_DIR / "hello.py"

if not sample_file.exists():
    sample_file.write_text(
        'print("Hello Mazhar, from the sandbox!")\n'
    )


# ============================================================
# 4. Manifest
# ============================================================

manifest = Manifest(
    entries={
        "code": LocalDir(src=CODE_DIR)
    },
    extra_path_grants=[
        SandboxPathGrant(path=str(OUTPUT_DIR))
    ]
)


# ============================================================
# 5. Capabilities
# ============================================================

capabilities = Capabilities.default()


# ============================================================
# 6. Sandbox configuration
# ============================================================

run_config = RunConfig(
    sandbox=SandboxRunConfig(
        client=UnixLocalSandboxClient()
    ),
    workflow_name="Groq Sandbox Demo"
)


# ============================================================
# 7. Sandbox Agent
# ============================================================

agent = SandboxAgent(
    name="Code Assistant",

    instructions=f"""
You are a simple coding assistant.

Inspect the Python files inside the code directory.

Create a file called summary.txt inside:

{OUTPUT_DIR}

The summary should briefly explain what the Python
file does.

Do not modify the original Python file.
""",

    model=MODEL,

    default_manifest=manifest,
    capabilities=capabilities,
)


# ============================================================
# 8. Run
# ============================================================

async def main():

    result = await Runner.run(
        agent,
        "Inspect the code directory and create the summary file.",
        run_config=run_config,
    )

    print("\nAgent response:")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())  




    # It automatically creates a code folder containing a sample Python file and an output folder. 
    # The Manifest tells the agent which folders it can access, while Capabilities define what it can do in the sandbox. 
    # The agent is then instructed to read the Python file and create a summary in the output folder. 
    # Finally, Runner.run() starts the agent, allowing the LLM to reason about the task 
    # and work with the files inside its controlled sandbox environment.

    # ImportError: UnixLocalSandbox is not supported on Windows.
    # Solution: Use a Unix-based system (Linux or macOS) to run this code, as the UnixLocalSandboxClient is not compatible with Windows.
    # Solution: Docker -> use DockerSandboxClient instead of UnixLocalSandboxClient.