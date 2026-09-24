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
