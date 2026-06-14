import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import yaml

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Safety Co-Pilot — Vertical Moving Bridge",
    layout="wide",
    page_icon="🛡️",
)

# ----------------------------
# File helpers
# ----------------------------


def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.environ.get(name, default)


# ----------------------------
# Graph helpers
# ----------------------------


def fact(subject, relation, object_, source="design", confidence=1.0, evidence="") -> Dict[str, Any]:
    return {
        "subject": str(subject),
        "relation": str(relation),
        "object": str(object_),
        "source": source,
        "confidence": float(confidence),
        "evidence": str(evidence),
    }


def dedupe_graph(graph: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for f in graph:
        key = (f["subject"], f["relation"], f["object"], f.get("source", ""))
        if key not in seen:
            out.append(f)
            seen.add(key)
    return out


def slug(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "entity"


def graph_terms(graph: List[Dict[str, Any]]) -> set:
    terms = set()
    for f in graph:
        terms.add(f["subject"])
        terms.add(f["relation"])
        terms.add(f["object"])
    return terms


def ontology_synonym_map(ontology: Dict[str, Any]) -> Dict[str, str]:
    mapping = {}
    for cid, cdef in ontology.get("concepts", {}).items():
        mapping[cid.lower()] = cid
        mapping[cdef.get("canonical_name", cid).lower()] = cid
        for syn in cdef.get("synonyms", []) or []:
            mapping[str(syn).lower()] = cid
    return mapping


def canonicalize_concept(value: str, ontology: Dict[str, Any]) -> str:
    if not value:
        return value
    raw = str(value).strip()
    mapping = ontology_synonym_map(ontology)
    if raw.lower() in mapping:
        return mapping[raw.lower()]
    raw_slug = slug(raw)
    if raw_slug in ontology.get("concepts", {}):
        return raw_slug
    return raw_slug


def add_ontology_facts(graph: List[Dict[str, Any]], ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
    concepts = ontology.get("concepts", {})
    enriched = list(graph)

    # Add properties and implied facts for every is_a concept.
    for f in list(graph):
        if f["relation"] != "is_a":
            continue
        entity = f["subject"]
        ctype = f["object"]
        cdef = concepts.get(ctype, {})
        for prop in cdef.get("properties", []) or []:
            enriched.append(fact(entity, "has_property", prop, "ontology", 0.85, f"{ctype}.properties"))
        for implied in cdef.get("implied_facts", []) or []:
            enriched.append(
                fact(
                    entity,
                    implied.get("relation", "has_property"),
                    implied.get("object", ""),
                    "ontology",
                    0.85,
                    f"{ctype}.implied_facts",
                )
            )

    return dedupe_graph(enriched)


# ----------------------------
# LLM normalization
# ----------------------------


def compact_ontology_for_prompt(ontology: Dict[str, Any], max_chars: int = 9000) -> str:
    rows = []
    for cid, cdef in ontology.get("concepts", {}).items():
        syns = ", ".join(cdef.get("synonyms", [])[:8])
        props = ", ".join(cdef.get("properties", [])[:12])
        rows.append(f"- {cid}: synonyms=[{syns}], properties=[{props}]")
    text = "\n".join(rows)
    return text[:max_chars]


def allowed_relations_for_prompt(relations: Dict[str, Any]) -> List[str]:
    return list(relations.get("relations", {}).keys())


def chat_json(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def llm_normalize_description(description: str, ontology: Dict[str, Any], relations: Dict[str, Any], api_key: str, model: str) -> Dict[str, Any]:
    system_prompt = """
You are a safety engineering extraction assistant.
Your job is to normalize an early design description into structured facts.
Do not decide final hazards. Extract only entities, properties, relations and unknowns.
Use the allowed ontology concepts and allowed relations when possible.
Return valid JSON only.
""".strip()

    user_prompt = f"""
DESIGN DESCRIPTION:
{description}

ONTOLOGY HINTS:
{compact_ontology_for_prompt(ontology)}

ALLOWED RELATIONS:
{allowed_relations_for_prompt(relations)}

Return JSON in this exact shape:
{{
  "entities": [
    {{
      "id": "short_snake_case_id",
      "type": "ontology_concept_or_clear_generic_type",
      "properties": ["property_or_concept"],
      "evidence": "short phrase from description"
    }}
  ],
  "relations": [
    {{
      "subject": "entity_id",
      "relation": "allowed_relation",
      "object": "entity_id_or_concept",
      "evidence": "short phrase from description"
    }}
  ],
  "unknowns": [
    "safety-relevant missing information from the description"
  ]
}}

Rules:
- Keep IDs stable and short.
- Prefer ontology concepts such as vertical_reciprocating_conveyor, bridge_platform, guide_rail, electric_motor, motor_controller, hazardous_voltage_supply, landing_gate, presence_sensor, emergency_stop, passenger, platform_edge, plastic_material, moisture.
- Extract hazards only as properties if explicitly or strongly implied; do not invent final risk decisions.
- If passengers/staff/operators can be near the bridge, include a human actor and human_access property.
- If 220/230 V AC or mains is present, classify the source as hazardous_voltage_supply.
- If motion is vertical, include vertical_motion and hazardous_movement properties.
""".strip()

    return chat_json(api_key, model, system_prompt, user_prompt, temperature=0.0)


def fallback_normalize_description(description: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback for UI testing without API key."""
    t = description.lower()
    entities = []
    relations = []
    unknowns = []

    def ent(eid, typ, props, evidence):
        entities.append({"id": eid, "type": typ, "properties": props, "evidence": evidence})

    ent("bridge_system", "vertical_moving_passenger_bridge", [], "system description")

    if any(w in t for w in ["vertical", "lift", "reciprocating", "moves up", "moves down"]):
        ent("vertical_lift_mechanism", "vertical_reciprocating_conveyor", ["moving_part", "vertical_motion", "hazardous_movement", "crushing_zone", "shearing_zone"], "vertical moving/lift mechanism")
        relations.append({"subject": "vertical_lift_mechanism", "relation": "connected_to", "object": "bridge_platform", "evidence": "lift moves platform"})

    if any(w in t for w in ["platform", "bridge"]):
        ent("bridge_platform", "bridge_platform", ["human_access", "elevated", "carries_persons"], "bridge platform")

    if any(w in t for w in ["passenger", "staff", "operator", "person", "human"]):
        ent("passenger", "passenger", ["human", "human_access"], "passengers/staff can be near bridge")
        relations.append({"subject": "passenger", "relation": "can_access", "object": "bridge_platform", "evidence": "passengers/staff access platform"})

    if any(w in t for w in ["edge", "gap", "entrance"]):
        ent("platform_edge", "platform_edge", ["fall_edge", "human_access"], "platform edge or transition")
        relations.append({"subject": "passenger", "relation": "near", "object": "platform_edge", "evidence": "passenger near edge"})

    if any(w in t for w in ["guide rail", "guide rails", "rail", "column"]):
        ent("guide_rails", "guide_rail", ["fixed_structure", "pinch_point"], "guide rails near moving platform")
        relations.append({"subject": "guide_rails", "relation": "near", "object": "vertical_lift_mechanism", "evidence": "guide rails near movement"})

    if any(w in t for w in ["motor", "actuator"]):
        ent("electric_motor", "electric_motor", ["moving_part", "mechanical_energy", "possible_ignition_source"], "motor/actuator drives lift")
        relations.append({"subject": "electric_motor", "relation": "drives", "object": "vertical_lift_mechanism", "evidence": "motor drives movement"})

    if any(w in t for w in ["controller", "control cabinet", "motor controller", "drive controller"]):
        ent("motor_controller", "motor_controller", ["electrical_energy", "control_function", "possible_ignition_source"], "motor controller")
        relations.append({"subject": "motor_controller", "relation": "controls", "object": "electric_motor", "evidence": "controller controls motor"})

    if any(w in t for w in ["230 v", "230v", "220 v", "220v", "mains", "ac electrical"]):
        ent("mains_supply", "hazardous_voltage_supply", ["hazardous_voltage", "electrical_energy"], "230/220 V or mains supply")
        relations.append({"subject": "mains_supply", "relation": "supplies_energy_to", "object": "motor_controller", "evidence": "mains supplies controller"})

    if any(w in t for w in ["plastic", "cable tray", "cover"]):
        ent("plastic_covers", "plastic_material", ["combustible_material"], "plastic covers/cable trays")
        relations.append({"subject": "plastic_covers", "relation": "near", "object": "motor_controller", "evidence": "plastic near electrical cabinet"})

    if any(w in t for w in ["gate", "door", "barrier"]):
        ent("landing_gate", "landing_gate", ["guard", "protective_measure"], "landing gate")
        relations.append({"subject": "landing_gate", "relation": "protects_from", "object": "platform_edge", "evidence": "gate intended to protect edge"})
        relations.append({"subject": "landing_gate", "relation": "protects_from", "object": "vertical_lift_mechanism", "evidence": "gate intended to protect moving zone"})

    if any(w in t for w in ["sensor", "presence", "obstruction", "light curtain"]):
        ent("presence_sensor", "presence_sensor", ["protective_device", "detection_function"], "presence/obstruction sensor")
        relations.append({"subject": "presence_sensor", "relation": "monitors", "object": "vertical_lift_mechanism", "evidence": "sensor monitors movement area"})

    if any(w in t for w in ["emergency stop", "e-stop", "estop"]):
        ent("emergency_stop", "emergency_stop", ["safety_control", "stop_function"], "emergency stop")
        relations.append({"subject": "emergency_stop", "relation": "stops", "object": "vertical_lift_mechanism", "evidence": "emergency stop provided"})

    if any(w in t for w in ["rain", "moisture", "water", "outdoor", "weather"]):
        ent("moisture_environment", "moisture", ["moisture"], "rain/moisture exposure")

    for phrase in ["not yet", "not defined", "unknown", "does not yet define"]:
        if phrase in t:
            unknowns.append("Some protection, control failure or verification details are not yet defined.")
            break

    return {"entities": entities, "relations": relations, "unknowns": unknowns}


def build_graph_from_normalized(normalized: Dict[str, Any], ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
    graph = []
    graph.append(fact("system", "is_a", "vertical_moving_passenger_bridge", "system", 1.0, "PoC domain"))

    for e in normalized.get("entities", []) or []:
        eid = slug(e.get("id", e.get("type", "entity")))
        typ = canonicalize_concept(e.get("type", "unknown_entity"), ontology)
        evidence = e.get("evidence", "")
        graph.append(fact("system", "has_component", eid, "llm_extraction", 0.9, evidence))
        graph.append(fact(eid, "is_a", typ, "llm_extraction", 0.9, evidence))
        for p in e.get("properties", []) or []:
            graph.append(fact(eid, "has_property", canonicalize_concept(p, ontology), "llm_extraction", 0.85, evidence))

    allowed_relation_names = set(load_yaml(DATA_DIR / "relations.yaml").get("relations", {}).keys())
    for r in normalized.get("relations", []) or []:
        rel = r.get("relation", "")
        if rel not in allowed_relation_names:
            continue
        graph.append(
            fact(
                slug(r.get("subject", "unknown_subject")),
                rel,
                slug(r.get("object", "unknown_object")),
                "llm_extraction",
                0.85,
                r.get("evidence", ""),
            )
        )

    return add_ontology_facts(dedupe_graph(graph), ontology)


# ----------------------------
# Propagation rules
# ----------------------------


def apply_propagation_rules(graph: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    terms = graph_terms(graph)
    derived = []
    for rule in rules.get("rules", []) or []:
        required = [c.get("concept_or_property") for c in rule.get("if_all", [])]
        if required and all(req in terms for req in required):
            d = rule.get("derive", {})
            derived.append(
                fact(
                    d.get("subject", "system_context"),
                    d.get("relation", "may_cause"),
                    d.get("object", rule.get("id", "derived_hazard_context")),
                    "propagation_rule",
                    0.75,
                    rule.get("id", ""),
                )
            )
    return dedupe_graph(graph + derived)


# ----------------------------
# Deterministic FTA template evaluation
# ----------------------------


def contains_any(graph: List[Dict[str, Any]], terms: List[str]) -> Tuple[bool, List[str]]:
    all_terms = graph_terms(graph)
    hits = sorted([t for t in terms if t in all_terms])
    return bool(hits), hits


def evaluate_node(node: Dict[str, Any], graph: List[Dict[str, Any]]) -> Dict[str, Any]:
    ntype = node.get("type", "EVENT")

    if ntype == "ASSUMED":
        return {
            "matched": True,
            "status": "assumed",
            "label": node.get("label", ""),
            "evidence": ["Assumed for generic early screening."],
            "missing_information": [],
        }

    if ntype == "UNKNOWN_OK":
        return {
            "matched": True,
            "status": "unknown_ok",
            "label": node.get("label", ""),
            "evidence": ["Not confirmed; treated as missing information for early screening."],
            "missing_information": node.get("missing_information", []),
        }

    if ntype == "EVENT":
        concepts = node.get("match", {}).get("concepts_any", [])
        ok, hits = contains_any(graph, concepts)
        return {
            "matched": ok,
            "status": "matched" if ok else "not_found",
            "label": node.get("label", ""),
            "evidence": hits,
            "missing_information": node.get("missing_information", []) if not ok else [],
        }

    if ntype in ["AND", "OR"]:
        child_results = [evaluate_node(c, graph) for c in node.get("children", []) or []]
        if ntype == "AND":
            required = [r for r in child_results if r["status"] not in ["unknown_ok", "assumed"]]
            matched_count = sum(1 for r in required if r["matched"])
            if required and matched_count == len(required):
                status = "potential_pathway"
                ok = True
            elif matched_count > 0:
                status = "partial_evidence"
                ok = False
            else:
                status = "no_evidence"
                ok = False
            return {
                "matched": ok,
                "status": status,
                "label": node.get("label", ""),
                "children": child_results,
                "evidence": [e for r in child_results for e in r.get("evidence", [])],
                "missing_information": [m for r in child_results for m in r.get("missing_information", [])],
            }

        ok = any(r["matched"] for r in child_results)
        return {
            "matched": ok,
            "status": "matched" if ok else "not_found",
            "label": node.get("label", ""),
            "children": child_results,
            "evidence": [e for r in child_results for e in r.get("evidence", [])],
            "missing_information": [m for r in child_results for m in r.get("missing_information", [])],
        }

    return {"matched": False, "status": "unknown_node_type", "label": node.get("label", ""), "evidence": [], "missing_information": []}


def collect_branch_summary(result: Dict[str, Any]) -> List[str]:
    rows = []

    def walk(r, level=0):
        prefix = "  " * level
        label = r.get("label", "")
        status = r.get("status", "")
        evidence = ", ".join(r.get("evidence", [])[:6])
        rows.append(prefix + f"- {label}: {status}" + (f" ({evidence})" if evidence else ""))
        for child in r.get("children", []) or []:
            walk(child, level + 1)

    walk(result)
    return rows


def evaluate_templates(graph: List[Dict[str, Any]], templates: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for t in templates.get("templates", []) or []:
        result = evaluate_node(t.get("fault_tree", {}), graph)
        if result["status"] in ["potential_pathway", "partial_evidence"]:
            evidence = sorted(set(result.get("evidence", [])))
            missing = sorted(set(result.get("missing_information", [])))
            results.append(
                {
                    "Template ID": t.get("id"),
                    "Hazard template": t.get("title"),
                    "Top event": t.get("top_event"),
                    "Category": t.get("category"),
                    "Deterministic status": result["status"],
                    "Deterministic confidence": "medium" if result["status"] == "potential_pathway" else "low",
                    "Deterministic evidence": ", ".join(evidence[:15]) if evidence else "No direct evidence.",
                    "Missing information from template": "; ".join(missing[:20]),
                    "Careful wording": t.get("output", {}).get("careful_wording", ""),
                    "Branch summary": "\n".join(collect_branch_summary(result)),
                    "Template object": t,
                }
            )
    return results


# ----------------------------
# LLM backward investigation
# ----------------------------


def compact_graph_for_prompt(graph: List[Dict[str, Any]], max_facts: int = 120) -> str:
    rows = []
    for f in graph[:max_facts]:
        evidence = f.get("evidence", "")
        rows.append(f"- {f['subject']} --{f['relation']}--> {f['object']} [source={f.get('source','')}; evidence={evidence[:80]}]")
    return "\n".join(rows)


def compact_template_for_prompt(template: Dict[str, Any]) -> str:
    return yaml.dump(template, sort_keys=False, allow_unicode=True)


def llm_backward_investigation(
    description: str,
    graph: List[Dict[str, Any]],
    template: Dict[str, Any],
    deterministic_row: Dict[str, Any],
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    system_prompt = """
You are a safety engineering reviewer performing backward hazard investigation.
You start from a generic hazard template and check whether the design description and graph support this hazard pathway.
You must be careful: do not claim a hazard will occur. Distinguish confirmed context, potential pathway, partial evidence, missing information only, or dismissed by evidence.
Return valid JSON only.
""".strip()

    user_prompt = f"""
ORIGINAL DESIGN DESCRIPTION:
{description}

SAFETY CONTEXT GRAPH FACTS:
{compact_graph_for_prompt(graph)}

GENERIC HAZARD TEMPLATE:
{compact_template_for_prompt(template)}

DETERMINISTIC TEMPLATE MATCH RESULT:
{json.dumps({k: v for k, v in deterministic_row.items() if k != 'Template object'}, indent=2)}

Investigate the hazard template backward:
- Which required ingredients are supported by the design?
- Which ingredients are only weakly supported?
- Which information is missing?
- Which developer questions should be asked?
- Is this a new potential hazard, changed hazard, or not supported?

Return JSON with this exact shape:
{{
  "status": "confirmed_context | strong_potential | potential_pathway | partial_evidence | missing_information_only | dismissed_by_evidence",
  "confidence_score": 0.0,
  "rationale": "short but specific explanation",
  "supported_ingredients": ["..."],
  "weak_or_assumed_ingredients": ["..."],
  "missing_information": ["..."],
  "questions_to_developer": ["..."],
  "possible_design_implications": ["..."],
  "do_not_claim": ["..."]
}}
""".strip()

    return chat_json(api_key, model, system_prompt, user_prompt, temperature=0.0)


def fallback_backward(row: Dict[str, Any]) -> Dict[str, Any]:
    status = "potential_pathway" if row["Deterministic status"] == "potential_pathway" else "partial_evidence"
    return {
        "status": status,
        "confidence_score": 0.55 if status == "potential_pathway" else 0.35,
        "rationale": row.get("Careful wording", "Deterministic template evidence indicates a possible hazard pathway."),
        "supported_ingredients": [row.get("Deterministic evidence", "")],
        "weak_or_assumed_ingredients": [],
        "missing_information": [m.strip() for m in row.get("Missing information from template", "").split(";") if m.strip()],
        "questions_to_developer": [m.strip() for m in row.get("Missing information from template", "").split(";") if m.strip()][:6],
        "possible_design_implications": ["Review design controls and update safety analysis if this hazard pathway is relevant."],
        "do_not_claim": ["Do not claim the hazard will occur; treat it as early design awareness."],
    }


# ----------------------------
# Graph visualization
# ----------------------------


def make_dot(graph: List[Dict[str, Any]]) -> str:
    interesting = {
        "has_component",
        "is_a",
        "has_property",
        "drives",
        "controls",
        "near",
        "can_access",
        "protects_from",
        "supplies_energy_to",
        "may_cause",
        "creates_hazard_zone",
    }
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        "  graph [fontname=Arial, bgcolor=white];",
        "  node [shape=box, style=rounded, fontname=Arial, fontsize=10];",
        "  edge [fontname=Arial, fontsize=9];",
    ]
    for f in graph:
        if f["relation"] in interesting:
            s = f["subject"].replace('"', '')
            o = f["object"].replace('"', '')
            r = f["relation"].replace('"', '')
            lines.append(f'  "{s}" -> "{o}" [label="{r}"];')
    lines.append("}")
    return "\n".join(lines)


# ----------------------------
# Streamlit UI
# ----------------------------


st.title("🛡️ Safety Co-Pilot — Vertical Moving Bridge")
st.caption("LLM normalization + deterministic Safety Context Graph + generic small FTA templates + LLM backward hazard investigation")

ontology = load_yaml(DATA_DIR / "ontology.yaml")
relations = load_yaml(DATA_DIR / "relations.yaml")
rules = load_yaml(DATA_DIR / "propagation_rules.yaml")
templates = load_yaml(DATA_DIR / "fault_tree_hazard_templates.yaml")
default_description = (DATA_DIR / "default_design_description.txt").read_text(encoding="utf-8")

with st.sidebar:
    st.header("LLM settings")
    default_key = safe_secret("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API key", value=default_key, type="password", help="Can also be set in Streamlit secrets.")
    model = st.text_input("Model", value="gpt-4o-mini")
    use_llm_normalization = st.checkbox("Use LLM for design normalization", value=bool(api_key))
    use_llm_backward = st.checkbox("Use LLM for backward hazard investigation", value=bool(api_key))
    investigate_scope = st.selectbox("Backward investigation scope", ["Deterministic matches only", "All templates"], index=0)

    st.divider()
    st.write("Loaded artifacts:")
    st.write("`ontology.yaml`")
    st.write("`relations.yaml`")
    st.write("`propagation_rules.yaml`")
    st.write("`fault_tree_hazard_templates.yaml`")

st.subheader("1. System / design description")
description = st.text_area("Unstructured design description", value=default_description, height=280)

run = st.button("Proceed: Analyze design", type="primary")

if not run:
    st.info("Edit the design description, add your OpenAI key if needed, then press Proceed.")
    st.stop()

# Step 1: LLM or fallback normalization
with st.spinner("Normalizing design description..."):
    if use_llm_normalization and api_key:
        try:
            normalized = llm_normalize_description(description, ontology, relations, api_key, model)
            normalization_source = "LLM"
        except Exception as e:
            st.warning(f"LLM normalization failed; using deterministic fallback. Error: {e}")
            normalized = fallback_normalize_description(description, ontology)
            normalization_source = "fallback"
    else:
        normalized = fallback_normalize_description(description, ontology)
        normalization_source = "fallback"

# Step 2: Build graph + deterministic reasoning
graph = build_graph_from_normalized(normalized, ontology)
graph = apply_propagation_rules(graph, rules)
deterministic_results = evaluate_templates(graph, templates)

# Optionally include all templates in backward analysis by creating partial placeholder rows
if investigate_scope == "All templates":
    existing_ids = {r["Template ID"] for r in deterministic_results}
    for t in templates.get("templates", []) or []:
        if t.get("id") not in existing_ids:
            deterministic_results.append(
                {
                    "Template ID": t.get("id"),
                    "Hazard template": t.get("title"),
                    "Top event": t.get("top_event"),
                    "Category": t.get("category"),
                    "Deterministic status": "no_evidence",
                    "Deterministic confidence": "low",
                    "Deterministic evidence": "No deterministic match.",
                    "Missing information from template": "",
                    "Careful wording": t.get("output", {}).get("careful_wording", ""),
                    "Branch summary": "No deterministic match.",
                    "Template object": t,
                }
            )

# Step 3: LLM backward analysis
backward_rows = []
with st.spinner("Running backward hazard investigation..."):
    for row in deterministic_results:
        template = row["Template object"]
        if use_llm_backward and api_key:
            try:
                b = llm_backward_investigation(description, graph, template, row, api_key, model)
            except Exception as e:
                b = fallback_backward(row)
                b["rationale"] = f"LLM backward investigation failed; fallback used. Error: {e}. " + b.get("rationale", "")
        else:
            b = fallback_backward(row)

        backward_rows.append(
            {
                "Hazard template": row["Hazard template"],
                "Template ID": row["Template ID"],
                "Deterministic status": row["Deterministic status"],
                "Backward status": b.get("status", "unknown"),
                "Confidence score": b.get("confidence_score", 0.0),
                "Rationale": b.get("rationale", ""),
                "Supported ingredients": "; ".join(b.get("supported_ingredients", []) or []),
                "Weak / assumed ingredients": "; ".join(b.get("weak_or_assumed_ingredients", []) or []),
                "Missing information": "; ".join(b.get("missing_information", []) or []),
                "Questions to developer": "; ".join(b.get("questions_to_developer", []) or []),
                "Possible design implications": "; ".join(b.get("possible_design_implications", []) or []),
                "Do not claim": "; ".join(b.get("do_not_claim", []) or []),
                "Branch summary": row.get("Branch summary", ""),
            }
        )

st.success(f"Analysis complete. Normalization source: {normalization_source}.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Result: backward hazard analysis",
    "Deterministic FTA matches",
    "Normalized facts",
    "Safety Context Graph",
    "Generic hazard templates",
])

with tab1:
    st.header("LLM backward hazard investigation")
    if backward_rows:
        df = pd.DataFrame(backward_rows)
        st.dataframe(
            df[[
                "Hazard template",
                "Backward status",
                "Confidence score",
                "Rationale",
                "Missing information",
                "Questions to developer",
                "Possible design implications",
            ]],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Detailed investigation")
        for item in backward_rows:
            with st.expander(f"{item['Hazard template']} — {item['Backward status']}"):
                st.markdown("**Rationale**")
                st.write(item["Rationale"])
                st.markdown("**Supported ingredients**")
                st.write(item["Supported ingredients"] or "—")
                st.markdown("**Weak / assumed ingredients**")
                st.write(item["Weak / assumed ingredients"] or "—")
                st.markdown("**Questions to developer**")
                st.write(item["Questions to developer"] or "—")
                st.markdown("**Do not claim**")
                st.write(item["Do not claim"] or "—")
                st.markdown("**Deterministic branch summary**")
                st.code(item["Branch summary"])
    else:
        st.info("No templates selected for backward investigation.")

with tab2:
    st.header("Deterministic generic FTA matches")
    if deterministic_results:
        df_det = pd.DataFrame([{k: v for k, v in r.items() if k != "Template object"} for r in deterministic_results])
        st.dataframe(
            df_det[[
                "Hazard template",
                "Deterministic status",
                "Deterministic confidence",
                "Deterministic evidence",
                "Missing information from template",
                "Careful wording",
                "Template ID",
            ]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No deterministic FTA hazard template matched.")

with tab3:
    st.header("Normalized facts")
    st.caption("This is the LLM/fallback extraction before deterministic ontology enrichment and propagation.")
    st.json(normalized)

with tab4:
    st.header("Safety Context Graph")
    st.graphviz_chart(make_dot(graph), use_container_width=True)
    st.subheader("Graph triples")
    st.dataframe(pd.DataFrame(graph), use_container_width=True, hide_index=True)

with tab5:
    st.header("Generic small FTA hazard templates")
    st.code(yaml.dump(templates, sort_keys=False, allow_unicode=True), language="yaml")
