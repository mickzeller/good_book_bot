import os
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")

if not HF_API_TOKEN:
    print("Warning: No Hugging Face token found. Some models may not work.")

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


def main():
    """
    Main function to set up and run the scripture RAG demo.
    """
    print("Starting the Scripture RAG Demo...")

    # scripture text should be placed in /data/*.txt
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        print("Please create it and add your scripture .txt files.")
        return

    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            print(f"Loading document: {filename}")
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())

    if not documents:
        print("No documents were loaded. Exiting.")
        return

    print(f"Loaded {len(documents)} document sections.")

    # chunking documents to allow text to be embedded into the vector store
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    if not any(doc.page_content.strip() for doc in documents):
        print("Warning: Documents appear to be empty.")

    print(f"Split into {len(docs)} chunks.")

    # create Embeddings and Vector Store
    print("Creating embeddings and building FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("Vector store created successfully.")

    # Use a smaller, more compatible model for local inference
    print("Loading language model...")
    model_name = "distilgpt2"

    # Create a text generation pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.half()  # Use FP16 to reduce memory

    # Set pad_token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
    )

    # Create LangChain LLM wrapper
    llm = HuggingFacePipeline(pipeline=pipe)

    # prompt template
    system_prompt = (
        "You are a helpful assistant answering questions about scripture. "
        "Use only the provided context to answer. If the context doesn't "
        "contain relevant information, say 'I cannot find that information "
        "in the provided text.'\n\nContext: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # creating RAG(Retrieval-Augmented Generation) chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
    print("QA Chain is ready.")

    question = "What did Jesus teach about forgiveness?"
    print(f"\nAsking question: {question}")

    try:
        result = qa_chain.invoke({"input": question})

        print("\n--- LLM Response ---")
        print(result["answer"])
        print("\n--- Retrieved Source Documents ---")
        for doc in result["context"]:
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Content: {doc.page_content[:250]}...\n")

    except Exception as e:
        print(f"Error during QA chain execution: {e}")
        print("Please check that your documents are properly formatted and accessible.")


if __name__ == "__main__":
    main()
