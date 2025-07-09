import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# Note: need to setup LLM from HuggingFace to use this
# from langchain_community.llms import HuggingFaceEndpoint
from langchain.llms.fake import FakeListLLM


def main():
    """
    Main function to set up and run the scripture RAG demo.
    """
    print("Starting the Scripture RAG Demo...")

    # scripture text should be placed in /data/*.txt
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please create it and add your scripture .txt files.")
        return

    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            print(f"Loading document: {filename}")
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())

    if not documents:
        print("No documents were loaded. Exiting.")
        return

    print(f"Loaded {len(documents)} document sections.")

    # chunking documents to allow text to be embedded into the vector store
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # create Embeddings and Vector Store
    print("Creating embeddings and building FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Vector store created successfully.")

    # IMPORTANT: This is a placeholder LLM. It doesn't use AI.
    # To use a real model, you would uncomment the HuggingFaceEndpoint import,
    # set up an API token, and replace FakeListLLM.
    # This just lets us test the retrieval part of RAG
    responses = ["Based on the context, the answer is...", "I found this relevant passage..."]
    llm = FakeListLLM(responses=responses)

    # prompt template
    prompt_template = """
    Use the following context to answer the question at the end.
    If you don't know the answer, just say that you don't know.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # creating RAG(Retrieval-Augmented Generation) chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    print("QA Chain is ready.")

    # asking a question
    question = "What is the purpose of baptism?"
    print(f"\nAsking question: {question}")

    result = qa_chain.invoke({"query": question})

    print("\n--- LLM Response ---")
    print(result["result"])
    print("\n--- Retrieved Source Documents ---")
    for doc in result["source_documents"]:
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Content: {doc.page_content[:250]}...\n")


if __name__ == "__main__":
    main()
