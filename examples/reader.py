"""Turn a file into LlamaIndex documents with NDI (Nace Document Intelligence)."""

from llama_index.readers.nace import NaceReader

documents = NaceReader().load_data("report.pdf", extra_info={"origin": "demo"})
for document in documents:
    print(document.metadata["file_name"], len(document.text), "characters")

# One document per page instead of one per file:
pages = NaceReader(chunking_strategy="page").load_data("report.pdf")
print(len(pages), "pages")
