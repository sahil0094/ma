Streaming protocol in the context of MLFlow ResponseAgent, Langgraph, and Databricks SDK
Created by: HO, Tam, last updated on: 19 Jan 2026 • 2 minute read

Table of Contents
Overview

Component Interaction

Event Type

ResponseTextDeltaEvent

ResponseOutputItemDoneEvent

ResponseCompletedEvent

ResponseErrorEvent

Overview
Smart Investigator is a multi-agent chatbot that is built using Langgraph, deployed by MLFlow Response Agent, and managed by Databricks Mosaic AI service. Within a conversation, multiple types of events are streamed to the frontend which will result in different behavior on the UI. This document aims to explain the purpose and the sequence of these events.

The structure of all events mentioned in this document can be found at: https://confluence.suncorp.app/confluence/x/3qiiZ

Component Interaction
Within the context of Smart Investigator, we have the following dynamics:

A Langgraph's graph that implements the business logic.

An object inheriting from MLFlow ResponseAgent is equipped with a predict_stream() method.

Within the predict_stream, a stream is tapped into the Langgraph's graph and captures the mode of "values", "custom", and "message". The "interrupt" is also captured via an exception mechanic. Each will yield a different ResponseAgentStreamEvent.

A streaming session to a ResponseAgent endpoint hosted by Databricks Mosaic AI can be conducted either through:

httpx_sse: The event returned will be a json-string in the structure of MLFlow ResponseAgentStreamEvent.

Databricks SDK: The event returned will be automatically cast into OpenAI ResponseAgent objects which share a great deal of similarities with their MLFlow twins. However, they are not identical.

In the context of this document, we will assume the Databricks SDK is used for streaming.

When the streaming is invoked to a ResponseAgent endpoint, the predict_stream method is called, which will in turn trigger a stream into the Langgraph's graph.

Event Type
ResponseTextDeltaEvent
This type of event will occur when an agent streams from the AzureOpenAI client within a Langgraph's graph as follows:

Python

llm_stream = []
for event in llm.stream("Tell me a joke"):
    llm_stream.append(event)
In this case, the Langgraph graph will yield a tuple of ("message", (chunk: langchain_core.messages.ai.AIMessageChunk, metadata: dict)). If the developer wants to stream this chunk out, it will be cast into a ResponseTextDeltaEvent.

Important note: If there is a stream of ResponseTextDeltaEvent going out, there must be a ResponseOutputItemDoneEvent with item.custom_outputs['event'] == "llm" sent at the end of the stream of ResponseTextDeltaEvent. This ResponseOutputItemDoneEvent should contain the whole text of the ResponseTextDeltaEvent stream and the name of the agent that sends it.

ResponseOutputItemDoneEvent
This type of event will occur when a Langgraph graph:

Finishes streaming from an AzureOpenAI client:

a. graph.stream yields mode="message".

b. ResponseOutputItemDoneEvent will have item.custom_outputs['event'] == "llm".

Reaches the END:

a. graph.stream yields mode="values".

b. ResponseOutputItemDoneEvent will have item.custom_outputs['event'] == "response".

Gets a writer in Langgraph that writes something:

a. graph.stream yields mode="custom".

b. ResponseOutputItemDoneEvent will have item.custom_outputs['event'] == "thinking".

The purpose of this event is to carry the actionable information/content between agent-agent or frontend-agent.

ResponseCompletedEvent
This type of event happens at the end of a streaming session to signify the completion unless a ResponseErrorEvent happens. This event also carries the state of the interacting agent in case it does not have the capability to maintain a persistent memory.

ResponseErrorEvent
This type of event happens when the interacting agent catches an exception.
