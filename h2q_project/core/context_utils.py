from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from h2q_project.utils.llm import get_llm

def get_project_context(file_path):
    """Generates a project context summary for a given file."""
    loader = TextLoader(file_path)
    documents = loader.load()

    # Use a more efficient text splitting strategy to reduce tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20, length_function=len)
    texts = text_splitter.split_documents(documents)

    llm = get_llm()

    # Use 'stuff' chain for summarization as it's simpler and often sufficient for smaller contexts
    chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
    summary = chain.run(texts)

    return summary
