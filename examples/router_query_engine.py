"""Route a query between two query engines with Drex instead of an LLM."""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.selectors.nace import NaceMultiSelector, NaceSingleSelector

# RouterQueryEngine resolves Settings.llm (OpenAI by default) when it is built, even if it
# only routes. MockLLM keeps this example key-free. In a real app, pass the LLM that writes
# answers: Drex replaces the routing decision, not answer generation.
llm = MockLLM()
embed_model = MockEmbedding(embed_dim=8)


def make_tool(name: str, description: str, text: str) -> QueryEngineTool:
    index = VectorStoreIndex.from_documents([Document(text=text)], embed_model=embed_model)
    return QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(llm=llm), name=name, description=description
    )


tools = [
    make_tool("pricing", "Plans, prices, invoices and refunds", "The Pro plan costs $20."),
    make_tool("docs", "API reference and error codes", "A 429 means you hit the rate limit."),
    make_tool("status", "Current outages and incident history", "No incidents today."),
]

QUERY = "Why do I keep getting 429 errors from the API?"


def show(title: str, selector: NaceSingleSelector | NaceMultiSelector) -> None:
    router = RouterQueryEngine(selector=selector, query_engine_tools=tools, llm=llm)
    response = router.query(QUERY)  # routes with Drex; MockLLM stands in for answer writing
    result = response.metadata["selector_result"]
    print(f"{title}: {[tools[s.index].metadata.name for s in result.selections]}")
    print("  probabilities:", result.probabilities)
    print("  fell_back:", result.fell_back)


show("single", NaceSingleSelector(min_confidence=0.5, default_index=1))
show("multi", NaceMultiSelector(threshold=0.5, max_outputs=2))
