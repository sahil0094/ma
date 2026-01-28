The system streams tokens in real-time from sub-agents to the frontend, NOT collecting the complete response first.

  ---
  How It Works in tool_factory.py

  1. Stream Opening (lines 74-95)
  _open_stream() → Returns a generator of events from:
    - Local agent: self.agent.predict_stream(request)
    - Databricks endpoint: self.client.responses.create(..., stream=True)

  2. Real-time Processing (lines 97-169)

  ┌─────────────────────────────────────────────────────────────────┐
  │                    _get_final_event() Loop                       │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  for event in stream:                                           │
  │      │                                                          │
  │      ├── event.type == 'response.output_text.delta'             │
  │      │       │                                                  │
  │      │       └──→ writer(llm_event)  ← IMMEDIATE STREAM         │
  │      │                                  (token by token)        │
  │      │                                                          │
  │      ├── event.type == 'response.output_item.done'              │
  │      │       │                                                  │
  │      │       ├── event_type == 'llm' or 'thinking'              │
  │      │       │       └──→ writer(llm_event)  ← IMMEDIATE STREAM │
  │      │       │                                                  │
  │      │       └── event_type == 'response'                       │
  │      │               └──→ final_response = event  ← STORE ONLY  │
  │      │                                                          │
  │      └── event.type == 'response.completed'                     │
  │              └──→ state = event.response.metadata  ← EXTRACT    │
  │                                                                  │
  │  return final_response, state                                   │
  └─────────────────────────────────────────────────────────────────┘

  ---
  Event Types & Streaming Behavior
  ┌──────────────────────────────────────────┬────────────────────────────┬─────────────────────────────┐
  │                Event Type                │          Behavior          │    When Sent to Frontend    │
  ├──────────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ response.output_text.delta               │ Token-by-token LLM output  │ Immediately via writer()    │
  ├──────────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ response.output_item.done (llm/thinking) │ Completed thinking message │ Immediately via writer()    │
  ├──────────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ response.output_item.done (response)     │ Final agent response       │ Stored, returned after loop │
  ├──────────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ response.completed                       │ State metadata             │ Extracted, not streamed     │
  └──────────────────────────────────────────┴────────────────────────────┴─────────────────────────────┘
  ---
  Data Flow Diagram

  ┌─────────────┐     stream=True      ┌─────────────────┐
  │  Sub-Agent  │ ──────────────────→  │  Databricks     │
  │  (Feature)  │                      │  Mosaic Endpoint│
  └─────────────┘                      └────────┬────────┘
                                                │
                                                │ SSE Events
                                                ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      tool_factory.py                             │
  │                                                                  │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │  _get_final_event()                                      │   │
  │  │                                                          │   │
  │  │  Token 1 ──→ writer() ──→ Frontend (displays "H")       │   │
  │  │  Token 2 ──→ writer() ──→ Frontend (displays "He")      │   │
  │  │  Token 3 ──→ writer() ──→ Frontend (displays "Hel")     │   │
  │  │  Token 4 ──→ writer() ──→ Frontend (displays "Hell")    │   │
  │  │  Token 5 ──→ writer() ──→ Frontend (displays "Hello")   │   │
  │  │  ...                                                     │   │
  │  │  Final  ──→ stored ──→ return as ToolMessage            │   │
  │  └──────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────┘
                                                │
                                                │ StreamWriter
                                                ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      response_agent.py                           │
  │                                                                  │
  │  predict_stream() yields events to Databricks serving layer     │
  └─────────────────────────────────────────────────────────────────┘
                                                │
                                                │ SSE to client
                                                ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                         FRONTEND                                 │
  │                                                                  │
  │  Receives and displays tokens as they arrive                    │
  └─────────────────────────────────────────────────────────────────┘

  ---
  Key Code Evidence

  Line 115-118: Delta events streamed immediately
  if event.type == 'response.output_text.delta':
      ml_event = ResponsesAgentStreamEvent(delta=event.delta, ...)
      llm_event = prepare_thinking_message(stream_event=ml_event)
      writer(llm_event)  # ← Immediate write to frontend

  Line 139-140: Thinking/LLM done events streamed immediately
  llm_event = prepare_thinking_message(stream_event=ml_event)
  writer(llm_event)  # ← Immediate write to frontend

  Line 141-142: Final response stored, not streamed
  elif event_type == DoneType.response:
      final_response = event  # ← Stored, returned later

  ---
  Summary
  ┌───────────────────────────────────────────┬───────────────────────────────────────────────┐
  │                 Question                  │                    Answer                     │
  ├───────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ Is it streaming chunk by chunk?           │ Yes - tokens stream in real-time              │
  ├───────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ Is it collecting complete response first? │ No - only the final response event is stored  │
  ├───────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ What streams immediately?                 │ delta tokens, thinking messages, llm messages │
  ├───────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ What is returned after streaming?         │ Final response event + state metadata         │
  └───────────────────────────────────────────┴───────────────────────────────────────────────┘
  The StreamWriter from LangGraph allows the tool to emit events during execution, which propagate up through the graph to predict_stream() and out to the frontend via Server-Sent Events (SSE).
