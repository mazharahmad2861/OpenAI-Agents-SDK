#  OpenAI Agents SDK 
## Lab 1

### Introduction to building AI agents using the **OpenAI Agents SDK**, with a focus on asynchronous execution, tools, tracing, streaming, and memory.

## Topics Covered

### AsyncIO

* **`async def`** — defines an asynchronous function.
* **`await`** — waits for an asynchronous operation to complete.
* **`asyncio.run()`** — starts and runs an asynchronous program.
* **`asyncio.gather()`** — runs independent asynchronous tasks concurrently.
* **Event Loop** — manages and schedules asynchronous tasks.
* **Coroutine** — the asynchronous operation created by calling an `async def` function.
* AsyncIO is mainly useful for **I/O-bound tasks** such as API and LLM calls.

### Agents SDK

* **Agent** — combines a model, instructions, and tools.
* **Runner** — executes an agent and manages its application-level turn.
* **`final_output`** — provides the agent's final response.
* **Tracing** — provides observability into agent execution.
* **Streaming** — allows execution events to be received as they occur.

### Tools

* **Function Tools** — expose Python functions to an agent.
* **`@function_tool`** — converts a Python function into an agent tool.
* **Type hints** — define tool parameters.
* **Docstrings** — describe the tool to the model.
* The LLM decides when a tool should be called.

### Agent Collaboration

* **Agents as Tools** — one agent can use another agent as a tool.
* **Handoffs** — transfer a task to another specialized agent.
* **Guardrails** — control and validate agent inputs and outputs.

### Sessions & Memory

* Each `Runner.run()` call is normally a fresh start.
* **Manual memory** — preserve conversation history using `to_input_list()`.
* **`SQLiteSession`** — built-in session-based conversation memory.
* Sessions allow multiple agent runs to share conversation history.
* Memory can be kept in-memory or stored in a SQLite database.

### Model Provider

The project uses **Groq's OpenAI-compatible API** with:

* **Model:** `openai/gpt-oss-20b`
* **Provider:** Groq
* **API style:** OpenAI-compatible Chat Completions

## Overall Flow

```text
AsyncIO
   ↓
Agent
   ↓
Runner
   ↓
Model
   ↓
Tools
   ↓
Response
```

With additional capabilities:

```text
Agent
├── Tools
├── Streaming
├── Tracing
├── Guardrails
├── Handoffs
└── Sessions / Memory
```

## Key Takeaway

This session establishes the core building blocks required to create **asynchronous, tool-using, observable, and stateful AI agents** using the OpenAI Agents SDK.
## Lab 2 & 3 
# Agentic AI — Multi-Agent Orchestration & Controls

A practical exploration of building **agentic AI workflows** using the OpenAI Agents SDK. This project focuses on multi-agent orchestration, structured outputs, guardrails, tool usage, email automation, sandbox execution, and MCP.

---

## 🚀 What This Project Covers

### 1. Agent Orchestration

**Orchestration** determines which agents run, in what order, and how the next step is decided.

Three approaches were explored:

* **Orchestration via Code** — Python explicitly controls the workflow using multiple `Runner.run()` calls. This approach is predictable, deterministic, and easy to debug.
* **Agents as Tools** — A manager agent dynamically invokes specialized agents as tools while retaining control of the overall workflow.
* **Handoffs** — An agent delegates the task to another specialist, transferring control to that agent.

The project demonstrates when to use deterministic code-based workflows versus more autonomous LLM-driven orchestration.

---

## 🤖 2. Multi-Agent Sales Automation

A practical **automated SDR (Sales Development Representative)** workflow was developed for sales outreach.

Specialized agents generate emails using different styles:

* Professional
* Humorous
* Executive / concise

Their outputs can then be:

1. Generated in parallel
2. Evaluated and selected
3. Refined
4. Sent through an email tool

This demonstrates how multiple specialized agents can collaborate to complete a business workflow.

---

## 🛠️ 3. Tools & Function Calling

Agents can be equipped with tools that allow them to perform real actions.

Examples include:

* Sending emails
* Calling specialized agents as tools
* Connecting to external capabilities

The `Agent.as_tool()` pattern allows an existing agent to be exposed as a tool that another agent can invoke.

This creates a manager → specialist architecture while keeping the manager in control.

---

## 📧 4. Email Automation

The project demonstrates sending emails through **SMTP**.

The workflow uses:

* SMTP server configuration
* Email credentials
* App-specific passwords
* Python's email/SMTP libraries

A **Pushover** notification fallback was also explored when email sending is unavailable.

The key idea is connecting an agent to a real-world action through a tool.

---

## 📦 5. Structured Outputs

LLMs normally return free-form text. Structured Outputs allow them to return predictable data based on a defined schema.

**Pydantic** is used to define the expected structure.

Example application: evaluating generated sales emails using fields such as:

* Professionalism
* Number of sentences
* Presence of placeholders

The result can then be used directly by Python logic instead of parsing natural-language responses.

### Flow

```text
LLM
 ↓
Structured JSON
 ↓
Pydantic Model
 ↓
Python Object
 ↓
Application Logic
```

The concept of **constrained decoding** was also introduced as a mechanism for enforcing structured output formats.

---

## 🛡️ 6. Guardrails

**Guardrails** are controls that prevent undesirable agent behavior.

Three types were explored:

* **Input Guardrails** — Validate incoming requests.
* **Output Guardrails** — Validate generated responses.
* **Tool Guardrails** — Validate tool execution.

A **tripwire** can stop execution when a guardrail detects a problem.

Guardrails were implemented both through the Agents SDK and through simpler explicit Python validation using a separate checker agent.

---

## 📂 7. Sandbox Agents

Sandbox Agents provide agents with a controlled execution environment.

They can work with files, modify code, execute commands, and generate artifacts within a defined workspace.

Three important concepts:

* **Manifest** — Defines which files/directories are accessible.
* **Capabilities** — Defines what operations the agent can perform.
* **SandboxRunConfig** — Defines the sandbox execution environment.

This enables more capable coding and file-manipulation agents while keeping their environment controlled.

---

## 🔌 8. Model Context Protocol (MCP)

**MCP (Model Context Protocol)** was introduced as a standardized way for agents to connect to external tools and sources of context.

Conceptually:

```text
Agent
 ↓
MCP Server
 ↓
External Tools / Context
 ↓
Agent
```

MCP extends an agent beyond its built-in capabilities and provides a standardized integration layer.

---

## 🏗️ Overall Architecture

```text
                         Agentic AI
                             │
          ┌──────────────────┼──────────────────┐
          ↓                  ↓                  ↓
   Orchestration      Structured Outputs     Guardrails
          │                  │                  │
    ┌─────┼─────┐            ↓                  ↓
    ↓     ↓     ↓         Pydantic          Validation
   Code  Tools Handoff        │              & Control
    │                          │
    └──────────┬───────────────┘
               ↓
        Multi-Agent Workflow
               │
       ┌───────┼────────┐
       ↓       ↓        ↓
     Email   Sandbox    MCP
```

---

## 🧰 Technologies

* Python
* OpenAI Agents SDK
* Pydantic
* AsyncIO
* SMTP
* Pushover
* Sandbox Agents
* Model Context Protocol (MCP)

---

## 🎯 Key Takeaways

* **Orchestration** controls how agents collaborate.
* **Code-based orchestration** provides predictability and control.
* **Agents as Tools** enable dynamic delegation while retaining manager control.
* **Handoffs** allow agents to transfer control to specialists.
* **Structured Outputs** convert LLM responses into reliable application objects.
* **Guardrails** add validation and behavioral controls.
* **Tools** allow agents to perform real-world actions.
* **Sandbox Agents** provide controlled execution environments.
* **MCP** enables standardized connections to external tools and context.

Overall, this project demonstrates how individual LLM calls can evolve into **structured, controlled, and capable multi-agent systems** suitable for real-world workflows. 
# Lab 4 & 5
# 🔎 Deep Research Agent

A multi-agent **Deep Research system** built with the **OpenAI Agents SDK**, **Groq `gpt-oss-20b`**, and **Tavily**.

The system takes a research question, plans multiple searches, executes them in parallel, generates a structured research report, and optionally sends the report by email.

---

## 🚀 Overview

This project demonstrates **orchestration by code**.

Instead of using one manager agent to decide the entire workflow, **Python explicitly controls the execution order** using multiple `Runner.run()` calls.

```text
User Query
    ↓
Planner Agent
    ↓
Search Agents × N
    ↓
Search Results
    ↓
Writer Agent
    ↓
Structured Report
    ↓
Email Agent
    ↓
Email / Notification
```

---

## 🤖 Agents

### 1. Search Agent

Responsible for researching individual search queries.

- Uses **Tavily** for web search
- Summarizes search results
- Produces concise research summaries

### 2. Planner Agent

Converts the user's research question into multiple search queries.

Uses **Structured Outputs** with Pydantic:

```text
WebSearchPlan
 └── WebSearchItem[]
      ├── reason
      └── query
```

The `reason` explains why each search is useful before generating the actual query.

### 3. Writer Agent

Combines the search results into a comprehensive report.

Returns:

```text
ReportData
 ├── short_summary
 ├── markdown_report
 └── follow_up_questions
```

### 4. Email Agent

Takes the generated report and prepares a professional email.

It uses a local `send_email_tool` to send the final report through SMTP, or a push notification when email is disabled.

---

## 🧠 Orchestration by Code

The core idea of the project is **code-based orchestration**.

Python controls the workflow:

```text
Planner
   ↓
Search
   ↓
Writer
   ↓
Email
```

The main workflow is essentially:

```text
run_searches()
      ↓
write_report()
      ↓
send_report_email()
```

Each stage is executed through `Runner.run()`.

This approach provides predictable execution because the workflow is explicitly defined in code.

---

## ⚡ Parallel Search

The planner creates multiple searches.

Instead of executing them one after another, the project uses:

```text
asyncio.gather()
```

Conceptually:

```text
Search 1 ──┐
Search 2 ──┤
Search 3 ──┼──→ Results
Search 4 ──┤
Search 5 ──┘
```

This allows independent searches to run concurrently.

---

## 📦 Structured Outputs

The project uses **Pydantic models** to make important agent outputs predictable.

### Search Plan

```text
WebSearchPlan
    ↓
WebSearchItem
    ├── reason
    └── query
```

### Research Report

```text
ReportData
    ├── short_summary
    ├── markdown_report
    └── follow_up_questions
```

Instead of relying entirely on free-form text, the application receives structured data that Python can reliably consume.

---

## 🔧 Technology Stack

| Technology | Purpose |
|---|---|
| OpenAI Agents SDK | Agent creation and orchestration |
| Groq `gpt-oss-20b` | LLM |
| Tavily | Web search |
| Pydantic | Structured outputs |
| asyncio | Parallel search execution |
| SMTP | Email delivery |
| Python | Application orchestration |
| dotenv | Environment configuration |

---



### `deep_research.py`

Contains:

- Agents
- Pydantic schemas
- Tavily search tool
- Email tool
- Orchestration functions
- Main workflow

### `messenger.py`

Contains the email and notification implementation.

### `.env`

Stores API keys and email configuration.

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

EMAIL_ADDRESS=your_email
EMAIL_APP_PASSWORD=your_app_password
RECIPIENTS=recipient@example.com
```

Keep `.env` out of version control.

---

## ▶️ Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python deep_research.py
```

The system will:

1. Receive the research question
2. Generate multiple search queries
3. Search the web using Tavily
4. Run searches concurrently
5. Generate a structured research report
6. Send the report through the Email Agent

---

## 🔄 Complete Workflow

```text
                    USER QUERY
                        │
                        ▼
                ┌──────────────┐
                │    PLANNER   │
                │   Groq 20B   │
                └──────┬───────┘
                       │
                 Search Plan
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Search        Search       Search
        Agent         Agent        Agent
          │            │            │
          └────────────┼────────────┘
                       │
                asyncio.gather()
                       │
                       ▼
                Search Results
                       │
                       ▼
                ┌──────────────┐
                │    WRITER    │
                │   Groq 20B   │
                └──────┬───────┘
                       │
                  ReportData
                       │
                       ▼
                ┌──────────────┐
                │    EMAIL     │
                │    AGENT     │
                └──────┬───────┘
                       │
                 Email Tool
                       │
                       ▼
                    EMAIL
```

---

## 🎯 Key Concepts Demonstrated

- Multi-agent systems
- Agent orchestration by code
- `Runner.run()`
- Structured Outputs
- Pydantic models
- Function tools
- Web search tools
- Parallel execution with `asyncio`
- Agent-to-agent data flow
- SMTP email automation
- Tracing with `trace()`
- Groq OpenAI-compatible API
- External tool integration with Tavily

---

## 💡 Why This Architecture?

The workflow is known in advance:

```text
Plan → Search → Write → Send
```

Therefore, Python handles the deterministic workflow while the LLM handles tasks requiring reasoning.

This makes the system easier to understand, debug, and control than giving a single manager agent responsibility for the entire workflow.

---

## 🔮 Possible Improvements

- Add source citations to the final report
- Add retry and error handling
- Add search result deduplication
- Add research quality evaluation
- Add configurable search depth
- Add persistent research history
- Add a web UI with Streamlit
- Add FastAPI endpoints
- Add human approval before sending emails
- Replace SMTP with a production email service
- Add observability and cost tracking
- Add alternative search providers
- Add more specialized research agents

---

## 📌 Core Takeaway

The main lesson of this project is:

> **Agents perform intelligent tasks; Python orchestrates the workflow.**

The project demonstrates how a complex Agentic AI application can be built from a small number of specialized agents connected through explicit code-based orchestration.
