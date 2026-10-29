# Examples

Set `NACE_API_KEY`, then `make examples` from the repo root runs the scripts.

| File | What it shows |
|---|---|
| `workflow_branching.py` | Support triage `Workflow`: `NaceSingleSelector` branches, low confidence goes to human review |
| `router_query_engine.py` | `RouterQueryEngine` with single and multi selectors (MockLLM for answers) |
| `evaluation.py` | Faithfulness, relevancy and correctness on a good and a bad answer |
| `selectors.ipynb` | Selectors tour, including the multi-select limitation |
| `evaluation.ipynb` | Evaluators tour, including images |

`RouterQueryEngine` needs an LLM at construction, so the examples pass `MockLLM()`. In a real app pass the LLM that writes answers.

Re-execute notebooks: `uv run jupyter nbconvert --execute --to notebook --inplace examples/*.ipynb`.
