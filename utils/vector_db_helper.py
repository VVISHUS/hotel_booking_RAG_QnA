import os
import chromadb
from chromadb.utils import embedding_functions
import torch
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

embedding_model = 'all-MiniLM-L6-v2'
persist_directory = "./chroma_db"
collection_name=os.getenv("chroma_db_collection")
# print(collection_name)
class VecDBHelper:
    def __init__(self, 
                 embedding_model=embedding_model, 
                 dir=persist_directory, 
                 device=device, 
                 collection_name=collection_name):
        
        self.collection = None

        try:
            self.embedding_model = embedding_model
            self.dir = dir
            self.device = device
            self.collection_name = collection_name
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model,
                device=self.device.type
            )
            self.chroma_client = chromadb.PersistentClient(path=self.dir)


            if self.collection_name:
                self._load_or_create_collection(self.collection_name)

            print("ChromaDB client initialized.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize VecDBHelper: {e}")

    def _load_or_create_collection(self, name):
        """Load if exists, else create."""
        existing_collections = self.chroma_client.list_collections()
        if any(c.name == name for c in existing_collections):
            print(f"Collection '{name}' exists. Loading...")
            self.collection = self.chroma_client.get_collection(
                name=name,
                embedding_function=self.embedding_function
            )
            count = self.collection.count()
            print(f"Collection loaded with {count} records.")
        else:
            print(f"No collection named '{name}' found. Creating new one.")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Collection '{name}' created successfully.")
            self.populate_vector_database(txt_file="utils/entity_definition.txt")

    def del_collection(self, collection_name=None):
        name = collection_name or self.collection_name
        if not name:
            print("[ERROR] No collection name provided.")
            return

        try:
            print(f"You're about to delete the collection: '{name}'")
            do_delete = input("Type 'Yes' to confirm: ")
            if do_delete.lower() == "yes":
                self.chroma_client.delete_collection(name=name)
                print(f"Collection '{name}' deleted successfully.")
                if self.collection_name == name:
                    self.collection = None
            else:
                print("Collection deletion canceled.")
        except Exception as e:
            print(f"[ERROR] Failed to delete collection: {e}")

    def populate_vector_database(self, txt_file):
        if not self.collection:
            print("[ERROR] Collection not initialized. Cannot populate.")
            return

        try:
            with open(txt_file, 'r') as f:
                content = f.read()

            sections = content.split('## ')[1:]
            analytics_docs = []
            analytics_ids = []
            analytics_metadatas = []

            for section in sections:
                lines = section.strip().split('\n')
                title = lines[0]

                table_name_line = next((line for line in lines if "Table Name" in line), "")
                table_name = table_name_line.split('`')[1] if '`' in table_name_line else ""

                description_line = next((line for line in lines if "Description" in line), "")
                description = description_line.split('Description: ')[1] if 'Description' in description_line else ""

                sample_questions = []
                found_questions = False
                for line in lines:
                    if "Sample Questions" in line:
                        found_questions = True
                        if ":" in line:
                            questions_text = line.split("Sample Questions: ")[1]
                            questions = [q.strip(' "\'') for q in questions_text.split('", "') if q.strip()]
                            sample_questions.extend(questions)
                        continue
                    if found_questions and "Key Metrics" in line:
                        break
                    if found_questions:
                        clean_line = line.strip(' -"\'')
                        if clean_line:
                            sample_questions.append(clean_line)

                key_metrics_line = next((line for line in lines if "Key Metrics" in line), "")
                key_metrics = key_metrics_line.split('Key Metrics: ')[1] if 'Key Metrics' in key_metrics_line else ""

                doc = f"""
                    Analytics: {title}
                    Table: {table_name}
                    Description: {description}
                    Sample Questions: {', '.join(sample_questions)}
                    Key Metrics: {key_metrics}
                """
                metadata = {
                    'record_type': 'analytics_definition',
                    'analytics_title': title,
                    'table_name': table_name,
                    'key_metrics': key_metrics
                }

                analytics_docs.append(doc)
                analytics_ids.append(table_name)
                analytics_metadatas.append(metadata)

            if analytics_docs:
                self.collection.add(
                    ids=analytics_ids,
                    documents=analytics_docs,
                    metadatas=analytics_metadatas
                )
                print(f"Added {len(analytics_docs)} analytics definitions to ChromaDB")

            total_count = self.collection.count()
            print(f"Total records in ChromaDB: {total_count}")
        except Exception as e:
            print(f"[ERROR] Failed to populate vector database: {e}")

    def get_query_results(self, query: str, top_k: int = 2):
        if not self.collection:
            print("[ERROR] Collection not initialized. Cannot query.")
            return []

        try:
            print(f"Fetching results for query: '{query}'...")
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            print(f"\n{results}\n")
            return results
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return []
