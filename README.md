# Safety Co-Pilot — Vertical Moving Bridge PoC

This Streamlit demo shows **Early Design Hazard Awareness** for a vertical moving passenger bridge / bridge mounted on a vertical reciprocating conveyor.

The demo is intentionally deterministic and generic:

- It builds a Safety Context Graph from a formal YAML design description.
- It applies ontology-derived properties.
- It evaluates generic small FTA-style hazard templates.
- It reports potential hazards, rationale, evidence, and missing information.

## Included hazard templates

The templates use only generic hazard definitions:

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
  example_vertical_moving_bridge_design.yaml
.streamlit/
  config.toml
```

## Important

This is an early design awareness demo. It does not certify safety and does not replace an expert risk assessment.
