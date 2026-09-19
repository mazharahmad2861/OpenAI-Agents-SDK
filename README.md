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
