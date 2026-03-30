from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from models import ClarificationRequest, SimulationOutput, SimulationParams
from tools import generate_metadata, parse_user_prompt, run_simulation, validate_params


class GraphState(TypedDict, total=False):
    user_prompt: str
    extracted_params: Optional[Dict[str, Any]]
    validated_params: Optional[SimulationParams]
    clarification_needed: bool
    clarification_questions: Optional[List[str]]
    clarification_response: Optional[str]
    generated_paths: Optional[List[str]]
    simulation_output: Optional[SimulationOutput]
    messages: List[str]


def _append_message(state: GraphState, text: str) -> List[str]:
    # Append a trace message to the running state log for observability in notebook demos.
    messages = list(state.get("messages", []))
    messages.append(text)
    return messages


def parse_prompt_node(state: GraphState) -> GraphState:
    # Parse the user's natural-language request into candidate simulation parameters.
    extracted = parse_user_prompt(state.get("user_prompt", ""))
    return {
        "extracted_params": extracted,
        "messages": _append_message(state, f"Parsed params: {extracted}"),
    }


def validate_node(state: GraphState) -> GraphState:
    # Validate parameters and decide whether to continue or request clarification.
    merged = dict(state.get("extracted_params") or {})

    response = state.get("clarification_response")
    if response:
        # Parse user clarification text and merge it over previously extracted fields.
        clarification_update = parse_user_prompt(response)
        merged.update(clarification_update)

    validation = validate_params(merged)

    if isinstance(validation, ClarificationRequest):
        # Route to human-in-the-loop clarification when required fields are missing/invalid.
        return {
            "extracted_params": merged,
            "clarification_needed": True,
            "clarification_questions": validation.questions,
            "messages": _append_message(state, "Clarification required."),
        }

    # Validation succeeded, so store final structured params.
    return {
        "extracted_params": merged,
        "validated_params": validation,
        "clarification_needed": False,
        "clarification_questions": None,
        "clarification_response": None,
        "messages": _append_message(state, f"Validated params: {validation.model_dump()}"),
    }


def clarify_node(state: GraphState) -> GraphState:
    # Pause execution and ask user clarification questions using LangGraph interrupt.
    payload = {
        "missing_fields": state.get("clarification_questions") or [],
        "questions": state.get("clarification_questions") or [],
        "instruction": "Please answer the missing details in one sentence.",
    }
    user_reply = interrupt(payload)
    return {
        "clarification_response": str(user_reply),
        "messages": _append_message(state, f"Received clarification: {user_reply}"),
    }


def simulate_node(state: GraphState) -> GraphState:
    # Run DeepLense simulation for validated parameters and collect output paths.
    params = state.get("validated_params")
    if params is None:
        raise ValueError("Simulation called without validated parameters.")

    if isinstance(params, dict):
        params = SimulationParams(**params)

    paths = run_simulation(params)
    return {
        "validated_params": params,
        "generated_paths": paths,
        "messages": _append_message(state, f"Generated {len(paths)} image(s)."),
    }


def output_node(state: GraphState) -> GraphState:
    # Build final typed simulation output payload for notebook display and downstream usage.
    params = state.get("validated_params")
    if params is None:
        raise ValueError("Output generation called without validated parameters.")

    if isinstance(params, dict):
        params = SimulationParams(**params)

    paths = state.get("generated_paths") or []
    metadata = generate_metadata(params, paths)

    return {
        "simulation_output": metadata,
        "messages": _append_message(state, "Simulation metadata generated."),
    }


def route_after_validate(state: GraphState) -> str:
    # Route to HITL clarification or simulation branch based on validation result.
    return "clarify" if state.get("clarification_needed", False) else "simulate"


def build_graph():
    # Build and compile the full LangGraph workflow requested for Specific Test II.
    builder = StateGraph(GraphState)

    builder.add_node("parse_prompt", parse_prompt_node)
    builder.add_node("validate", validate_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("simulate", simulate_node)
    builder.add_node("output", output_node)

    builder.add_edge(START, "parse_prompt")
    builder.add_edge("parse_prompt", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "clarify": "clarify",
            "simulate": "simulate",
        },
    )
    builder.add_edge("clarify", "validate")
    builder.add_edge("simulate", "output")
    builder.add_edge("output", END)

    return builder.compile()
