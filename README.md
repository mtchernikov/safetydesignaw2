# Safety Co-Pilot — Vertical Moving Bridge PoC with LLM Backward Analysis

This Streamlit demo mirrors the earlier lung ventilator / toaster flow:

1. **LLM normalization**: the LLM reads an unstructured design description and extracts entities, properties, relations and unknowns.
2. **Deterministic graph building**: the app builds a Safety Context Graph from normalized facts.
3. **Ontology enrichment**: generic safety properties and implied facts are added deterministically.
4. **Generic FTA hazard-template matching**: small AND/OR fault-tree templates are evaluated deterministically.
5. **LLM backward investigation**: for each relevant hazard template, the LLM starts from the hazard pattern and investigates whether the design graph supports, partially supports, or does not support the hazard path.

The hazard templates are deliberately generic:

- Electric shock
- Fire / overheating
- Mechanical injury due to moving parts
- Crushing / shearing / trapping
- Fall from height
- Collision / impact
- Loss of support / stability
- Unintended movement

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Main file path:

```text
app.py
```

Add your OpenAI API key in Streamlit Cloud secrets:

```toml
OPENAI_API_KEY="sk-..."
```

If no API key is available, the app uses a deterministic keyword fallback for normalization and a fallback backward analysis. That fallback is useful for testing the UI, but the intended PoC flow uses the LLM.

## Files

```text
app.py
requirements.txt
README.md

data/
  ontology.yaml
  relations.yaml
  propagation_rules.yaml
  fault_tree_hazard_templates.yaml
  default_design_description.txt

.streamlit/
  config.toml
  secrets.toml.example
```

## Important

This demo is for early design hazard awareness. It does not certify product safety and does not replace expert risk assessment.
