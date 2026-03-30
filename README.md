# GSoC 2026 ML4Sci DeepLense Tests

This repository contains solutions for Common Test I and Specific Test II.

## Project Structure

- `Common_Test_I/`
  - `deep_lense_resnet18.ipynb` - Main notebook for multi-class lensing image classification with ResNet18.
  - `resnet18_lensing.pth` - Trained model weights.
  - `roc_curve.png` - ROC curve plot from evaluation.
- `Specific_Test_II/`
  - `deeplense_agent.ipynb` - Main notebook for the agentic workflow using Pydantic AI + LangGraph.
  - `models.py` - Pydantic schemas for simulation inputs, outputs, and agent state.
  - `tools.py` - Prompt parsing, validation, DeepLense simulation runner, and metadata functions.
  - `graph.py` - LangGraph state machine with parse, validate, clarify (HITL), simulate, and output nodes.
  - `outputs/` - Generated gravitational lensing images.
- `.gitignore` - Excludes local environment files and secrets.

## Notes

- Set your OpenAI key in `Specific_Test_II/.env` before running the agent notebook.
- Install dependencies in your virtual environment before executing notebooks.
