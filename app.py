import yaml
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List, Tuple

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Safety Co-Pilot — Vertical Moving Bridge",
    layout="wide",
    page_icon="🛡️",
)

def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fact(subject, relation, object_, source="design", confidence=1.0, evidence=""):
    return {
        "subject": str(subject),
        "relation": str(relation),
        "object": str(object_),
        "source": source,
        "confidence": confidence,
        "evidence": evidence,
    }

def dedupe_graph(graph):
    seen = set()
    out = []
    for f in graph:
        key = (f["subject"], f["relation"], f["object"], f["source"])
        if key not in seen:
            out.append(f)
            seen.add(key)
    return out

def build_graph_from_design(design: Dict[str, Any], ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
    graph = []
    system = design.get("system", {})
    system_id = system.get("id", "system")
    system_type = system.get("type", "unknown_system")
    graph.append(fact(system_id, "is_a", system_type, "design", 1.0, "system.type"))

    for section in ["components", "materials", "environment"]:
        for item in design.get(section, []) or []:
            iid = item.get("id")
            itype = item.get("type", section[:-1])
            if not iid:
                continue
            graph.append(fact(system_id, "has_component", iid, "design", 1.0, f"{section} listed"))
            graph.append(fact(iid, "is_a", itype, "design", 1.0, f"{iid}.type"))
            for prop in item.get("properties", []) or []:
                graph.append(fact(iid, "has_property", prop, "design", 1.0, f"{iid}.properties"))
            for rel in ["connected_to", "near", "located_inside", "drives", "controls", "stops", "monitors", "protects_from", "supplies_energy_to"]:
                for target in item.get(rel, []) or []:
                    graph.append(fact(iid, rel, target, "design", 1.0, f"{iid}.{rel}"))

    concepts = ontology.get("concepts", {})
    for f in list(graph):
        if f["relation"] == "is_a":
            entity = f["subject"]
            ctype = f["object"]
            cdef = concepts.get(ctype, {})
            for prop in cdef.get("properties", []) or []:
                graph.append(fact(entity, "has_property", prop, "ontology", 0.85, f"{ctype}.properties"))
            for implied in cdef.get("implied_facts", []) or []:
                graph.append(fact(entity, implied.get("relation", "has_property"), implied.get("object", ""), "ontology", 0.85, f"{ctype}.implied_facts"))
    return dedupe_graph(graph)

def graph_terms(graph: List[Dict[str, Any]]) -> set:
    terms = set()
    for f in graph:
        terms.add(f["subject"])
        terms.add(f["relation"])
        terms.add(f["object"])
    return terms

def apply_propagation_rules(graph: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    terms = graph_terms(graph)
    derived = []
    for rule in rules.get("rules", []):
        required = [c.get("concept_or_property") for c in rule.get("if_all", [])]
        if required and all(req in terms for req in required):
            d = rule.get("derive", {})
            derived.append(fact("system_context", d.get("relation", "may_cause"), d.get("object", rule.get("id")), "propagation_rule", 0.75, rule.get("id", "")))
    return dedupe_graph(graph + derived)

def contains_any(graph, terms: List[str]) -> Tuple[bool, List[str]]:
    all_terms = graph_terms(graph)
    hits = sorted([t for t in terms if t in all_terms])
    return bool(hits), hits

def evaluate_node(node: Dict[str, Any], graph: List[Dict[str, Any]]) -> Dict[str, Any]:
    ntype = node.get("type", "EVENT")
    if ntype == "ASSUMED":
        return {"matched": True, "status": "assumed", "label": node.get("label", ""), "evidence": ["Assumed for generic early screening."], "missing_information": []}
    if ntype == "UNKNOWN_OK":
        return {"matched": True, "status": "unknown_ok", "label": node.get("label", ""), "evidence": ["Missing / not confirmed in design."], "missing_information": node.get("missing_information", [])}
    if ntype == "EVENT":
        concepts = node.get("match", {}).get("concepts_any", [])
        ok, hits = contains_any(graph, concepts)
        return {"matched": ok, "status": "matched" if ok else "not_found", "label": node.get("label", ""), "evidence": hits, "missing_information": node.get("missing_information", []) if not ok else []}
    if ntype in ["AND", "OR"]:
        child_results = [evaluate_node(c, graph) for c in node.get("children", [])]
        if ntype == "AND":
            required = [r for r in child_results if r["status"] not in ["unknown_ok", "assumed"]]
            ok = all(r["matched"] for r in required) if required else True
            return {"matched": ok, "status": "potential_pathway" if ok else "partial_evidence", "label": node.get("label", ""), "children": child_results, "evidence": [e for r in child_results for e in r.get("evidence", [])], "missing_information": [m for r in child_results for m in r.get("missing_information", [])]}
        ok = any(r["matched"] for r in child_results)
        return {"matched": ok, "status": "matched" if ok else "not_found", "label": node.get("label", ""), "children": child_results, "evidence": [e for r in child_results for e in r.get("evidence", [])], "missing_information": [m for r in child_results for m in r.get("missing_information", [])]}
    return {"matched": False, "status": "unknown_node_type", "label": node.get("label", ""), "evidence": [], "missing_information": []}

def collect_branch_summary(result: Dict[str, Any]) -> List[str]:
    rows = []
    def walk(r, level=0):
        if "children" not in r:
            evidence = ", ".join(r.get("evidence", [])[:6])
            rows.append("  " * level + f"- {r.get('label','')}: {r.get('status','')}" + (f" ({evidence})" if evidence else ""))
        else:
            rows.append("  " * level + f"- {r.get('label','')}: {r.get('status','')}")
            for c in r["children"]:
                walk(c, level + 1)
    walk(result)
    return rows

def evaluate_templates(graph: List[Dict[str, Any]], templates: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for t in templates.get("templates", []):
        eval_result = evaluate_node(t.get("fault_tree", {}), graph)
        if eval_result["matched"] or eval_result["status"] == "partial_evidence":
            evidence = sorted(set(eval_result.get("evidence", [])))
            missing = sorted(set(eval_result.get("missing_information", [])))
            status = eval_result["status"]
            confidence = "medium" if status == "potential_pathway" else "low"
            results.append({
                "Hazard template": t.get("title"),
                "Top event": t.get("top_event"),
                "Category": t.get("category"),
                "Status": status,
                "Confidence": confidence,
                "Why triggered": ", ".join(evidence[:12]) if evidence else "Template partly matched / missing information relevant.",
                "Missing information": "; ".join(missing[:12]),
                "Careful wording": t.get("output", {}).get("careful_wording", ""),
                "Template ID": t.get("id"),
                "Branch summary": "\n".join(collect_branch_summary(eval_result)),
            })
    return results

def make_dot(graph: List[Dict[str, Any]]) -> str:
    interesting = {"has_component", "is_a", "has_property", "drives", "controls", "near", "protects_from", "supplies_energy_to", "may_cause"}
    lines = ["digraph G {", "  rankdir=LR;", "  graph [fontname=Arial, bgcolor=white];", "  node [shape=box, style=rounded, fontname=Arial, fontsize=10];", "  edge [fontname=Arial, fontsize=9];"]
    for f in graph:
        if f["relation"] in interesting:
            s = f["subject"].replace('"', '')
            o = f["object"].replace('"', '')
            r = f["relation"].replace('"', '')
            lines.append(f'  "{s}" -> "{o}" [label="{r}"];')
    lines.append("}")
    return "\n".join(lines)

st.title("🛡️ Safety Co-Pilot — Vertical Moving Bridge")
st.subheader("Early Design Hazard Awareness with generic small FTA hazard templates")

ontology = load_yaml(DATA_DIR / "ontology.yaml")
rules = load_yaml(DATA_DIR / "propagation_rules.yaml")
templates = load_yaml(DATA_DIR / "fault_tree_hazard_templates.yaml")
example_text = (DATA_DIR / "example_vertical_moving_bridge_design.yaml").read_text(encoding="utf-8")

with st.sidebar:
    st.header("Files loaded")
    st.write("`ontology.yaml`")
    st.write("`propagation_rules.yaml`")
    st.write("`fault_tree_hazard_templates.yaml`")
    st.info("This demo uses generic hazard templates only. No product-specific standard mapping is included.")

description = st.text_area("Formal design description YAML", value=example_text, height=540)
run = st.button("Proceed: Build graph and evaluate hazards", type="primary")

if run:
    try:
        design = yaml.safe_load(description)
        if not isinstance(design, dict):
            raise ValueError("YAML root must be a dictionary.")
    except Exception as e:
        st.error(f"Could not parse YAML: {e}")
        st.stop()

    graph = build_graph_from_design(design, ontology)
    graph = apply_propagation_rules(graph, rules)
    hazard_results = evaluate_templates(graph, templates)

    st.success("Analysis complete.")
    tab1, tab2, tab3, tab4 = st.tabs(["Hazard results", "Safety Context Graph", "Graph triples", "Templates"])
    with tab1:
        st.header("Potential hazards identified")
        if hazard_results:
            df = pd.DataFrame(hazard_results)
            st.dataframe(df[["Hazard template", "Status", "Confidence", "Why triggered", "Missing information", "Careful wording", "Template ID"]], use_container_width=True, hide_index=True)
            st.subheader("Branch evidence")
            for item in hazard_results:
                with st.expander(f"{item['Hazard template']} — {item['Status']}"):
                    st.code(item["Branch summary"])
        else:
            st.info("No generic hazard template matched the current graph.")
    with tab2:
        st.header("Safety Context Graph")
        st.graphviz_chart(make_dot(graph), use_container_width=True)
    with tab3:
        st.header("Graph triples")
        st.dataframe(pd.DataFrame(graph), use_container_width=True, hide_index=True)
    with tab4:
        st.header("Generic FTA hazard templates")
        st.code(yaml.dump(templates, sort_keys=False, allow_unicode=True), language="yaml")
else:
    st.info("Edit the YAML if needed, then press Proceed.")
