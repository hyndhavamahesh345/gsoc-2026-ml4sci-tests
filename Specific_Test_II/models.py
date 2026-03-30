from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SimulationParams(BaseModel):
    model_type: Literal["Model_I", "Model_II"]
    substructure_type: Literal["no_sub", "subhalo"]
    num_images: int = Field(ge=1, le=100)
    source_redshift: float = Field(ge=0.1, le=3.0)
    lens_redshift: float = Field(ge=0.1, le=1.0)
    resolution: int = 64


class SimulationOutput(BaseModel):
    params_used: SimulationParams
    images_generated: int
    image_paths: List[str]
    timestamp: str
    status: str


class ClarificationRequest(BaseModel):
    missing_fields: List[str]
    questions: List[str]


class AgentState(BaseModel):
    user_prompt: str
    extracted_params: Optional[dict] = None
    validated_params: Optional[SimulationParams] = None
    clarification_needed: bool = False
    clarification_questions: Optional[List[str]] = None
    simulation_output: Optional[SimulationOutput] = None
    messages: List[str] = Field(default_factory=list)
