import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import traceback

class MarkdownHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_processed = 0
        
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.md'):
            current_time = time.time()
            if current_time - self.last_processed > 5:
                self.process_documents()
                self.last_processed = current_time
                
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.md'):
            self.process_documents()
            
    def process_documents(self):
        print("Processing document changes...")
        try:
            loader = DirectoryLoader(
                "/app/data/markdown",
                glob="**/*.md",
                loader_cls=TextLoader,
                silent_errors=True
            )
            
            docs = loader.load()
            print(f"Loaded {len(docs)} documents")
            
            if not docs:
                print("No documents to process")
                return
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            splits = text_splitter.split_documents(docs)
            print(f"Split into {len(splits)} chunks")
            
            # Validate API key
            nebius_api_key = os.getenv("NEBIUS_API_KEY")
            if not nebius_api_key:
                raise ValueError("NEBIUS_API_KEY environment variable is not set")
            
            embeddings = OpenAIEmbeddings(
                model="Qwen/Qwen3-Embedding-8B",
                openai_api_key=nebius_api_key,
                openai_api_base="https://api.studio.nebius.com/v1/",
                tiktoken_enabled=False
            )
            
            db = Chroma.from_documents(
                splits, 
                embeddings, 
                persist_directory="/app/chroma_data"
            )
   
            print("Document processing complete!")
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            traceback.print_exc()

def main():
    print("Starting document watcher...")
    
    handler = MarkdownHandler()
    handler.process_documents()
    
    observer = Observer()
    observer.schedule(handler, "/app/data/markdown", recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Stopping document watcher...")
    observer.join()

if __name__ == "__main__":
    main()