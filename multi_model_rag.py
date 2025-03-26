import os
import re
import uuid
import PIL
import fitz
import numpy as np
from PIL import Image
import io
import base64
import json
import hashlib
import pickle
import logging
import time
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
from utils import get_conversation_history,get_user_conversations
from utils import get_short_term_history, DB_CONFIG,store_short_term_message
import mysql.connector
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import redis
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Redis client for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Simple stopword list
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", 
    "of", "on", "that", "the", "to", "was", "were", "will", "with", "or", "but", "this", "there", "which"
}

class PDFProcessor:
    @staticmethod
    def extract_content(pdf_path: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        logger.info(f"Processing full PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        text_content = ""
        images = []
        tables = []
        pages_data = []
        keywords = set()

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            cleaned_text = PDFProcessor.clean_text(page_text)
            text_content += f"\n\n=== Page {page_num + 1} ===\n\n{cleaned_text}"
            pages_data.append({"page_number": page_num + 1, "text": cleaned_text})

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.width >= 100 and image.height >= 100:
                        buffered = io.BytesIO()
                        image.save(buffered, format=image.format if image.format else "PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        images.append({
                            "page": page_num,
                            "index": img_index,
                            "base64": img_base64,
                            "format": image.format if image.format else "PNG",
                            "width": image.width,
                            "height": image.height,
                            "size_bytes": len(image_bytes),
                            "extracted_path": f"{pdf_path}_page_{page_num}_img_{img_index}.png"
                        })
                    else:
                        logger.debug(f"Skipping small image on page {page_num + 1}: {image.width}x{image.height}")
                except Exception as e:
                    logger.error(f"Error processing image on page {page_num + 1}: {e}")

            lines = page.get_drawings()
            rectangles = [line["rect"] for line in lines if line["type"] == "re" and line["fill"] == 0]
            if len(rectangles) > 4:
                for i, rect in enumerate(rectangles):
                    if i > 0 and i % 4 == 0:
                        table_area = {
                            "page": page_num,
                            "rect": fitz.Rect(min(r.x0 for r in rectangles[i-4:i]), min(r.y0 for r in rectangles[i-4:i]), 
                                              max(r.x1 for r in rectangles[i-4:i]), max(r.y1 for r in rectangles[i-4:i])),
                            "text": PDFProcessor.clean_text(page.get_text("text", clip=fitz.Rect(min(r.x0 for r in rectangles[i-4:i]), 
                                                                                                  min(r.y0 for r in rectangles[i-4:i]), 
                                                                                                  max(r.x1 for r in rectangles[i-4:i]), 
                                                                                                  max(r.y1 for r in rectangles[i-4:i]))))
                        }
                        tables.append(table_area)

        logger.info(f"Extracted {len(images)} images from {pdf_path}")
        return text_content, images, tables, pages_data, list(keywords)

    @staticmethod
    def clean_text(text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Remove symbols and special characters (keep alphanumeric and spaces)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Remove consecutive dots
        text = re.sub(r'\.+', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove stopwords
        words = text.split()
        cleaned_words = [word for word in words if word not in STOPWORDS]
        text = ' '.join(cleaned_words)
        # Sanitize for sensitive data (e.g., email addresses)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED]', text)
        return text.strip()

# Intent classifier using a valid model from Hugging Face
try:
    intent_classifier = pipeline(
        "text-classification",
        model="Falconsai/intent_classification",
        tokenizer="Falconsai/intent_classification",
        framework="pt",
        top_k=None
    )
    logger.info("Successfully initialized intent classifier with model 'Falconsai/intent_classification'")
except Exception as e:
    logger.error(f"Failed to initialize intent classifier: {e}")
    # Fallback to a default intent classifier if the model fails to load
    intent_classifier = None

# Define intent mapping based on the model's expected labels
INTENT_MAPPING = {
    "statement": "informational",
    "question": "procedural",
    "clarification": "clarification",
    "out_of_domain": "out_of_domain",
    "greeting": "general",
    "image_request": "image_request"
}

def save_image_locally(image_data: Dict[str, Any], base_path: str) -> str:
    img_dir = os.path.join(base_path, "static", "images")
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"{uuid.uuid4()}.{image_data.get('format', 'png').lower()}")
    
    # Check if base64 exists, otherwise log an error and return a placeholder path
    if "base64" not in image_data:
        logger.error(f"Image data missing 'base64' key: {image_data}")
        return img_path  # Return path without saving; image won’t be accessible
    
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(image_data["base64"]))
    return img_path




    
class GeminiProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    async def process_pdf_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        images_data = []
        try:
            pdf_document = fitz.open(pdf_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_path = f"./extracted_images/{os.path.basename(pdf_path)}_page_{page_num}_img_{img_index}.{image_ext}"
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # Generate description using Gemini
                    image = PIL.Image.open(image_path)
                    prompt = "Describe this image in detail, focusing on elements relevant to cycle routes, maps, or design diagrams."
                    response = await asyncio.to_thread(self.model.generate_content, [prompt, image])
                    description = response.text.strip()

                    images_data.append({
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "extracted_path": image_path,
                        "description_json": {"description": description}
                    })
                    logger.info(f"Extracted image from {pdf_path}, page {page_num + 1}: {image_path}")
            pdf_document.close()
        except Exception as e:
            logger.error(f"Error processing images from {pdf_path}: {e}")
        return images_data

    async def process_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_bytes = base64.b64decode(image_data["base64"])
            prompt = """Describe this image in detail, focusing on road design elements like cycle lanes, pedestrian crossings, or width specifications. Format as JSON with:
            - "description": Detailed description
            - "key_elements": List of road-related features
            - "relevant_text": Extracted text (e.g., measurements)
            - "category": 'road_design' if applicable, else 'other'"""
            response = await asyncio.to_thread(self.model.generate_content, [prompt, {"mime_type": f"image/{image_data['format'].lower()}", "data": image_bytes}])
            description_text = response.text.strip()
            logger.debug(f"Gemini response for image on page {image_data['page'] + 1}: {description_text}")
            
            json_start, json_end = description_text.find('{'), description_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    description_json = json.loads(description_text[json_start:json_end])
                    required_keys = {"description": "", "key_elements": [], "relevant_text": [], "category": "unknown"}
                    for key, default in required_keys.items():
                        if key not in description_json:
                            description_json[key] = default
                            logger.warning(f"Missing key '{key}' in Gemini response, using default: {default}")
                    description_json["description"] = PDFProcessor.clean_text(description_json["description"])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Gemini response as JSON: {e}. Raw response: {description_text}")
                    description_json = {"description": PDFProcessor.clean_text(description_text), "key_elements": [], "relevant_text": [], "category": "unknown"}
            else:
                logger.warning(f"Gemini response not in JSON format: {description_text}")
                description_json = {"description": PDFProcessor.clean_text(description_text), "key_elements": [], "relevant_text": [], "category": "unknown"}

            return {
                "page": image_data["page"] + 1,
                "description_text": f"IMAGE DESCRIPTION (Page {image_data['page'] + 1}): {description_text}",
                "description_json": description_json,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": image_data["size_bytes"],
                "extracted_path": image_data["extracted_path"],
                "width": image_data["width"],
                "height": image_data["height"],
                "format": image_data["format"],
                "base64": image_data["base64"]  # Preserve the original base64 data
            }
        except Exception as e:
            logger.error(f"Error processing image with Gemini on page {image_data['page'] + 1}: {e}")
            return {
                "page": image_data["page"] + 1,
                "description_text": f"Failed: {str(e)}",
                "description_json": {"description": f"Error: {str(e)}", "key_elements": [], "relevant_text": [], "category": "error"},
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "base64": image_data.get("base64", "")  # Include base64 if present, empty string if not
            }
        

    async def process_table(self, table_data: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
        try:
            doc = fitz.open(pdf_path)
            page = doc[table_data["page"]]
            table_pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=table_data["rect"])
            img_bytes = table_pix.tobytes("png")
            prompt = """Extract tabular data in JSON with: "table_data" (2D array), "headers", "summary", "key_insights"."""
            response = await asyncio.to_thread(self.model.generate_content, [prompt, {"mime_type": "image/png", "data": img_bytes}])
            description_text = response.text
            json_start, json_end = description_text.find('{'), description_text.rfind('}') + 1
            table_json = json.loads(description_text[json_start:json_end]) if json_start >= 0 and json_end > json_start else {
                "table_data": [], "headers": [], "summary": description_text, "key_insights": []}
            return {
                "page": table_data["page"] + 1,
                "description_text": f"TABLE DESCRIPTION (Page {table_data['page'] + 1}): {description_text}",
                "table_json": table_json,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"Error processing table with Gemini: {e}")
            return {"page": table_data["page"] + 1, "description_text": f"Failed: {str(e)}", "table_json": {}, "error": str(e)}

class GlobalFAISSManager:
    def __init__(self, vector_db_path="./faiss_db/global_index", embedding_model=None):
        self.vector_db_path = vector_db_path
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        index_file = f"{self.vector_db_path}.faiss"
        if os.path.exists(index_file):
            self.vector_store = FAISS.load_local(folder_path=os.path.dirname(self.vector_db_path), index_name=os.path.basename(self.vector_db_path),
                                                 embeddings=self.embedding_model, allow_dangerous_deserialization=True)
            logger.info(f"Loaded Global FAISS index from {self.vector_db_path}")
        else:
            dummy_doc = Document(page_content="Initialization", metadata={"user_id": "global_init", "file_name": "init"})
            self.vector_store = FAISS.from_documents([dummy_doc], self.embedding_model)
            self.vector_store.save_local(folder_path=os.path.dirname(self.vector_db_path), index_name=os.path.basename(self.vector_db_path))

    async def add_user_embedding(self, user_id, content, metadata):
        try:
            document = Document(page_content=content, metadata=metadata | {"user_id": user_id, "file_name": metadata.get("file_name", "unknown")})
            await asyncio.to_thread(self.vector_store.add_documents, [document])
            await asyncio.to_thread(self.vector_store.save_local, folder_path=os.path.dirname(self.vector_db_path), index_name=os.path.basename(self.vector_db_path))
            logger.info(f"Added global memory for user {user_id}")
        except Exception as e:
            logger.error(f"Error adding to Global FAISS: {e}")

    async def search_user_memory(self, user_id, query, top_k=5):
        try:
            results = await asyncio.to_thread(self.vector_store.similarity_search, query, k=top_k)
            return [res for res in results if res.metadata.get("user_id") == user_id]
        except Exception as e:
            logger.error(f"Error searching global memory: {e}")
            return []

class VectorStoreManager:
    def __init__(self, vector_db_path: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.vector_db_path = vector_db_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
        self.documents = []
        self.bm25 = None
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        index_file = f"{self.vector_db_path}.faiss"
        if os.path.exists(index_file):
            self.vector_store = FAISS.load_local(folder_path=os.path.dirname(self.vector_db_path), index_name=os.path.basename(self.vector_db_path),
                                                 embeddings=self.embedding_model, allow_dangerous_deserialization=True)
            logger.info(f"Loaded FAISS index from {self.vector_db_path}")
        else:
            dummy_doc = Document(page_content="Initialization", metadata={"source": "init", "file_name": "init", "temporary": True})
            self.vector_store = FAISS.from_documents([dummy_doc], self.embedding_model)
            self.vector_store.save_local(folder_path=os.path.dirname(self.vector_db_path), index_name=os.path.basename(self.vector_db_path))

    async def add_documents(self, documents: List[Document]):
        if not documents:
            return
        try:
            for doc in documents:
                if "file_name" not in doc.metadata:
                    doc.metadata["file_name"] = "unknown"
            if len(self.vector_store.docstore._dict) == 1 and list(self.vector_store.docstore._dict.values())[0].metadata.get("temporary"):
                self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            else:
                await asyncio.to_thread(self.vector_store.add_documents, documents)
            await asyncio.to_thread(self.vector_store.save_local, folder_path=os.path.dirname(self.vector_db_path), index_name=os.path.basename(self.vector_db_path))
            self.documents.extend(documents)
            tokenized_docs = [doc.page_content.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"Added {len(documents)} documents to FAISS index")
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")

    def create_documents_from_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        chunks = self.text_splitter.split_text(text)
        documents = []
        current_page = 1
        for i, chunk in enumerate(chunks):
            page_match = re.search(r'=== Page (\d+) ===', chunk)
            if page_match:
                current_page = int(page_match.group(1))
            chunk_metadata = metadata.copy()
            chunk_metadata.update({"chunk_index": i, "page_number": current_page})
            if "file_name" not in chunk_metadata:
                chunk_metadata["file_name"] = "unknown"
            cleaned_chunk = re.sub(r'=== Page \d+ ===\n*', '', chunk).strip()
            documents.append(Document(page_content=cleaned_chunk, metadata=chunk_metadata))
        return documents

    async def rerank_results(self, query: str, documents: List[Document], cross_encoder: CrossEncoder, top_k: int = 5) -> List[Document]:
        if not documents:
            return []
        query_embedding = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)
        reranked = []
        for doc in documents:
            content = doc.page_content.lower()
            metadata = doc.metadata
            doc_embedding = np.array(self.embedding_model.embed_query(content)).reshape(1, -1)
            context_score = cosine_similarity(query_embedding, doc_embedding)[0][0]
            cross_encoder_score = cross_encoder.predict([[query, content]])
            technical_weight = 1.5 if "table" in content or metadata.get("priority_score", 0) > 1.0 else 1.0
            recency_score = 1.0 if "timestamp" not in metadata else max(0.5, 1.0 - (time.time() - metadata["timestamp"]) / (90 * 24 * 3600))
            final_score = (0.4 * context_score + 0.4 * cross_encoder_score + 0.1 * technical_weight + 0.1 * recency_score)
            reranked.append((final_score, doc))
        reranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in reranked[:top_k]]

    async def search_hybrid(self, query: str, k: int = 5, cross_encoder: CrossEncoder = None) -> List[Document]:
        # Dense retrieval
        dense_results = await asyncio.to_thread(self.vector_store.similarity_search, query, k=k * 2)
        dense_ids = {id(doc): doc for doc in dense_results}

        # Initialize sparse_ids as an empty dict to avoid undefined variable error
        sparse_ids = {}

        # Sparse retrieval with BM25
        if self.bm25:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_results = [(score, doc) for score, doc in zip(bm25_scores, self.documents)]
            bm25_results.sort(key=lambda x: x[0], reverse=True)
            sparse_ids = {id(doc): doc for score, doc in bm25_results[:k * 2]}

        # Combine results
        combined_ids = set(dense_ids.keys()) | set(sparse_ids.keys())
        combined_docs = [dense_ids.get(doc_id, sparse_ids.get(doc_id)) for doc_id in combined_ids]

        logger.debug(f"Initial search results for query '{query}': {len(combined_docs)} documents")
        if cross_encoder:
            reranked_results = await self.rerank_results(query, combined_docs, cross_encoder, top_k=k)
        else:
            reranked_results = combined_docs[:k]
        logger.debug(f"Reranked results for query '{query}': {len(reranked_results)} documents")
        return reranked_results



class ImageMetadataManager:
    def __init__(self, metadata_path: str = "./image_metadata.json", embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.metadata_path = metadata_path
        logger.debug(f"Initializing ImageMetadataManager with path: {metadata_path}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.metadata = self._load_metadata()
        logger.debug(f"Loaded {len(self.metadata)} metadata entries")
        self.image_vector_store = FAISS.from_documents(
            [Document(page_content="Initialization", metadata={"index": -1, "source": "init", "temporary": True})],
            self.embedding_model
        )
        self._build_image_vector_store()

    def _load_metadata(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                logger.debug(f"Metadata entries: {len(metadata)}")
                if not metadata:
                    logger.warning("Metadata file is empty")
                return metadata
        logger.warning(f"No image metadata file found at {self.metadata_path}")
        return []

    def _preprocess_text(self, text: str) -> str:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
        processed = " ".join(filtered_tokens)
        logger.debug(f"Preprocessed text: '{text}' -> '{processed}'")
        return processed

    def _build_image_vector_store(self):
        documents = []
        for idx, item in enumerate(self.metadata):
            if "description_json" in item and "description" in item["description_json"]:
                processed_desc = self._preprocess_text(item["description_json"]["description"])
                doc = Document(
                    page_content=processed_desc,
                    metadata={"index": idx, "source": item.get("source", ""), "temporary": False, "original_description": item["description_json"]["description"]}
                )
                documents.append(doc)
        if documents:
            self.image_vector_store = FAISS.from_documents(documents, self.embedding_model)
            logger.debug(f"Built image vector store with {len(documents)} documents")
        else:
            logger.warning("No documents to build image vector store")

    async def search_images(self, query: str, k: int = 2, cross_encoder: CrossEncoder = None, threshold: float = 0.05) -> List[Dict[str, Any]]:
        logger.debug(f"Searching images for query: {query}")
        processed_query = self._preprocess_text(query)
        query_embedding = np.array(self.embedding_model.embed_query(processed_query)).reshape(1, -1)
        results = await asyncio.to_thread(self.image_vector_store.similarity_search_with_score, processed_query, k=k * 2)
        image_metadata = []

        # Embedding-based search
        for doc, score in results:
            index = doc.metadata.get("index", -1)
            if index >= 0 and index < len(self.metadata):
                doc_embedding = np.array(self.embedding_model.embed_query(doc.page_content)).reshape(1, -1)
                similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
                if cross_encoder:
                    cross_encoder_score = cross_encoder.predict([[query, doc.metadata.get("original_description", doc.page_content)]])
                    similarity = (similarity + cross_encoder_score) / 2
                logger.debug(f"Image similarity score for query '{query}' and doc '{doc.page_content[:50]}...': {similarity}")
                if similarity >= threshold:
                    image_metadata.append((self.metadata[index], similarity))

        # Keyword-based fallback if embedding search fails
        if not image_metadata:
            logger.debug("Embedding search found no matches; falling back to keyword matching")
            query_tokens = set(processed_query.split())
            for idx, item in enumerate(self.metadata):
                if "description_json" in item and "description" in item["description_json"]:
                    desc = item["description_json"]["description"].lower()
                    desc_tokens = set(self._preprocess_text(desc).split())
                    common_tokens = query_tokens.intersection(desc_tokens)
                    if common_tokens:
                        logger.debug(f"Keyword match for query '{query}' and doc '{desc[:50]}...': Common tokens {common_tokens}")
                        image_metadata.append((self.metadata[idx], len(common_tokens) / len(query_tokens)))

        image_metadata.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Found {len(image_metadata)} images after embedding and keyword search")
        return [item[0] for item in image_metadata[:k]]
    

# def detect_intent(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, float]:
#         intents = {"general": 0.3, "image_request": 0.0, "table_request": 0.0, "follow_up": 0.0, "new_topic": 0.0}
#         combined = " ".join([msg["content"] for msg in history[-3:]] + [query]) if history else query
#         embedding = self.intent_model.encode(combined)
        
#         triggers = {
#             "image_request": ["show me", "diagram", "picture", "image", "look like"],
#             "table_request": ["table", "data", "chart", "summary"],
#             "follow_up": ["more", "explain", "what about", "details"],
#             "new_topic": ["let's start", "new topic", "switch to"]
#         }
#         for intent_key, trigger_set in triggers.items():
#             trigger_emb = self.intent_model.encode(" ".join(trigger_set))
#             similarity = cosine_similarity([embedding], [trigger_emb])[0][0]
#             intents[intent_key] = max(intents[intent_key], similarity if similarity > 0.6 else 0.0)
#         if max(intents.values()) < 0.6:
#             intents["general"] = 0.9
#         return intents

class MultiModalRAGManager:
    def __init__(self, groq_api_key: str, gemini_api_key: str, pdf_folder_path: str, vector_db_path: str = "./faiss_index",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", llm_model_name: str = "llama3-70b-8192",
                 json_output_dir: str = "./json_outputs", update_interval: int = 3600):
        self.pdf_folder_path = pdf_folder_path
        self.json_output_dir = json_output_dir
        self.checkpoint_file = os.path.join(os.path.dirname(vector_db_path), "ingestion_checkpoint.json")
        self.global_memory = GlobalFAISSManager()
        self.keyword_pool = {}
        self.keyword_window = 30  # Days
        self.update_interval = update_interval
        self.last_update_time = 0
        os.makedirs(self.json_output_dir, exist_ok=True)
        os.environ["GROQ_API_KEY"] = groq_api_key
        self.llm = ChatGroq(model_name=llm_model_name)
        self.gemini_processor = GeminiProcessor(api_key=gemini_api_key)
        self.text_vector_store = VectorStoreManager(vector_db_path=f"{vector_db_path}/text_index")
        self.image_metadata_manager = ImageMetadataManager()
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.processed_files_path = os.path.join(os.path.dirname(vector_db_path), "processed_files.pkl")
        self.processed_files = self._load_processed_files()
        self._load_checkpoint()
        self.session_expiry = 24 * 3600
        self.short_term_limit = 20
      

        

        self.rag_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are Atrip AI, specializing in road design, rules, and regulations. Provide concise, accurate answers, citing sources (e.g., 'Source: [Document Name], Page [Number]'). Use conversation history for context. Recent History: {history}\nDocument Context: {context}"""),
            ("human", "{question}")
        ])

        self.intent_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    async def initialize_session(self, user_id: str, conversation_id: str):
            session_key = f"session:{conversation_id}"
            if not redis_client.exists(session_key):
                redis_client.setex(session_key, self.session_expiry, json.dumps([]))
                await self.store_conversation_embedding(user_id, conversation_id, "system", "Session initialized", time.time())
            return session_key
    
    

    async def get_session_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        session_key = f"session:{conversation_id}"
        messages = redis_client.get(session_key)
        return json.loads(messages) if messages else []

    def _load_processed_files(self) -> Dict[str, str]:
        if os.path.exists(self.processed_files_path):
            try:
                with open(self.processed_files_path, 'rb') as f:
                    return {k.replace('\\', '/').lower(): v for k, v in pickle.load(f).items()}
            except Exception as e:
                logger.error(f"Error loading processed files: {e}")
        return {}
    
    async def validate_image(self, query: str, image_description: str) -> Dict[str, Any]:
        prompt = f"""
        Task: Determine if this image is relevant to the user's query. Be extremely lenient, accepting any image that might be even slightly related to road design, cycling infrastructure, pedestrian spaces, or maps that could include cycle routes or shared spaces.

        User Query: "{query}"
        Image Description: "{image_description}"

        Return: {{ "is_valid": true/false, "reason": "why or why not" }}
        """
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        try:
            validation = json.loads(response.content)
            logger.debug(f"Image validation result: {validation}")
            return validation
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation: {e}")
            return {"is_valid": False, "reason": "Validation parsing error"}

    def _save_processed_files(self):
        try:
            with open(self.processed_files_path, 'wb') as f:
                pickle.dump(self.processed_files, f)
        except Exception as e:
            logger.error(f"Error saving processed files: {e}")

    def detect_intent(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, float]:
        intents = {"general": 0.3, "image_request": 0.0, "table_request": 0.0, "follow_up": 0.0, "new_topic": 0.0}
        combined = " ".join([msg["content"] for msg in history[-3:]] + [query]) if history else query
        embedding = self.intent_model.encode(combined)
        
        triggers = {
            "image_request": ["show me", "diagram", "picture", "image", "look like", "design", "designs"],  # Added "design", "designs"
            "table_request": ["table", "data", "chart", "summary"],
            "follow_up": ["more", "explain", "what about", "details"],
            "new_topic": ["let's start", "new topic", "switch to"]
        }
        for intent_key, trigger_set in triggers.items():
            trigger_emb = self.intent_model.encode(" ".join(trigger_set))
            similarity = cosine_similarity([embedding], [trigger_emb])[0][0]
            intents[intent_key] = max(intents[intent_key], similarity if similarity > 0.5 else 0.0)  # Lowered threshold to 0.5
        if max(intents.values()) < 0.5:  # Lowered threshold
            intents["general"] = 0.9
        logger.debug(f"Intent detection details: {intents}")
        return intents

    def post_process_response(self, response: str) -> str:
        """Clean up the response text."""
        return response.strip()

    def add_follow_up_suggestions(self, response: str, question: str) -> str:
        """Add follow-up suggestions if relevant."""
        if "width" in question.lower():
            return f"{response}\n\nFor more details, you might ask: 'What are the minimum width requirements for cycle lanes in urban areas?'"
        return response

    def _get_file_hash(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {file_path}: {e}")
            raise

    def _load_checkpoint(self):
        self.checkpoint = {"last_file": "", "last_chunk": -1, "timestamp": 0}
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint: {self.checkpoint}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")

    def _save_checkpoint(self, file_path: str, chunk_index: int):
        self.checkpoint = {"last_file": file_path, "last_chunk": chunk_index, "timestamp": time.time()}
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint, f, indent=2)
            logger.debug(f"Saved checkpoint: {self.checkpoint}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def _save_json_data(self, file_path: str, data: Dict[str, Any]):
        json_file = os.path.join(self.json_output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.json")
        try:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved JSON data to {json_file}")
        except Exception as e:
            logger.error(f"Error saving JSON data for {file_path}: {e}")

    async def summarize_session(self, user_id: str, conversation_id: str):
        messages = await self.get_session_messages(conversation_id)
        if not messages:
            return "No conversation history to summarize."
        
        summary_prompt = f"Summarize this conversation:\n{json.dumps(messages)}"
        summary_response = await asyncio.to_thread(self.llm.invoke, summary_prompt)
        summary = summary_response.content.strip()
        
        # Store in MySQL
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_summaries (user_id, conversation_id, summary, timestamp) VALUES (%s, %s, %s, %s)",
            (user_id, conversation_id, summary, time.time())
        )
        conn.commit()
        cursor.close()
        conn.close()
        return summary

    def update_json_data(self):
        for pdf_path in self.processed_files.keys():
            json_file = os.path.join(self.json_output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._save_json_data(pdf_path, data)
                except Exception as e:
                    logger.error(f"Error updating JSON for {pdf_path}: {e}")

    async def ingest_single_document(self, file_path: str, filename: str, force_reprocess: bool = False) -> Dict[str, Any]:
        file_hash = self._get_file_hash(file_path)
        normalized_path = os.path.join(self.pdf_folder_path, filename).replace('\\', '/').lower()
        if not force_reprocess and normalized_path in self.processed_files and self.processed_files[normalized_path] == file_hash:
            logger.info(f"Skipping already processed file: {filename}")
            return {"skipped_files": 1, "processed_files": 0, "total_chunks": 0, "total_images": 0}

        start_chunk = self.checkpoint["last_chunk"] + 1 if self.checkpoint["last_file"] == normalized_path else 0
        text_content, images, tables, pages_data, keywords = PDFProcessor.extract_content(file_path)
        text_docs = self.text_vector_store.create_documents_from_text(text_content, {"source": file_path, "file_name": filename, "timestamp": time.time()})
        
        json_data = {
            "file_path": normalized_path,
            "text_content": text_content,
            "pages": pages_data,
            "tables": tables,
            "keywords": keywords,
            "last_processed": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for i, doc in enumerate(text_docs[start_chunk:], start=start_chunk):
            word_count = len(doc.page_content.split())
            doc.metadata["priority_score"] = sum(doc.page_content.lower().count(kw) for kw in keywords) / word_count if word_count > 0 else 0.0
            self._save_checkpoint(normalized_path, i)
            if i % 10 == 0:
                self._save_json_data(normalized_path, json_data)

        image_descriptions = await asyncio.gather(*[self.gemini_processor.process_image(img) for img in images])
        for img in image_descriptions:
            img["source"] = file_path
            img["local_path"] = save_image_locally(img, self.pdf_folder_path)  # Add local path
            await self.image_metadata_manager.add_image_metadata(img)
        
        json_data["image_descriptions"] = image_descriptions

        await self.text_vector_store.add_documents(text_docs[start_chunk:])
        self.processed_files[normalized_path] = file_hash
        self._save_processed_files()
        self._save_json_data(normalized_path, json_data)
        self._save_checkpoint(normalized_path, -1)
        
        return {"processed_files": 1, "skipped_files": 0, "total_chunks": len(text_docs), "total_images": len(image_descriptions)}

    async def ingest_documents(self, force_reprocess: bool = False) -> Dict[str, Any]:
        pdf_files = [os.path.join(root, file).replace('\\', '/').lower() for root, _, files in os.walk(self.pdf_folder_path) 
                     for file in files if file.lower().endswith('.pdf')]
        stats = {"total_files": len(pdf_files), "processed_files": 0, "skipped_files": 0, "total_chunks": 0, "total_images": 0}
        start_index = pdf_files.index(self.checkpoint["last_file"]) if self.checkpoint["last_file"] in pdf_files else 0
        
        for pdf_path in pdf_files[start_index:]:
            file_name = os.path.relpath(pdf_path, self.pdf_folder_path).replace('\\', '/')
            try:
                file_stats = await self.ingest_single_document(pdf_path, file_name, force_reprocess)
                stats["processed_files"] += file_stats["processed_files"]
                stats["skipped_files"] += file_stats["skipped_files"]
                stats["total_chunks"] += file_stats["total_chunks"]
                stats["total_images"] += file_stats["total_images"]
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                self._save_checkpoint(pdf_path, self.checkpoint["last_chunk"])
                raise
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self.update_json_data()
                self.last_update_time = current_time
        return stats

    async def get_relevant_history(self, user_id, conversation_id, question, history=None, short_term_limit=10, long_term_k=5):
        try:
            short_term = history if history is not None else get_short_term_history(conversation_id, limit=short_term_limit)
            global_results = await self.global_memory.search_user_memory(user_id, question, top_k=long_term_k)
            combined_history = [f"{msg['role']}: {msg['content']}" for msg in reversed(short_term[-5:])]
            question_embedding = self.embedding_model.embed_query(question)
            for doc in global_results:
                similarity = cosine_similarity(np.array([question_embedding]), 
                                               np.array([self.embedding_model.embed_query(doc.page_content)]))[0][0]
                combined_history.append(f"Past Memory: {doc.page_content} (Similarity: {similarity:.2f})")
            return "\n".join(combined_history[:10])
        except Exception as e:
            logger.error(f"Error fetching history: {e}")
            return ""

    async def store_conversation_embedding(self, user_id: str, conversation_id: str, role: str, content: str, timestamp: float):
        try:
            # Import inside the function to avoid circular import            
            # Call with only the 3 required arguments
            await asyncio.to_thread(store_short_term_message, conversation_id, role, content)
            
            # Store in global memory (FAISS)
            metadata = {"timestamp": timestamp, "conversation_id": conversation_id, "context": "text"}
            await self.global_memory.add_user_embedding(user_id, content, metadata)
            logger.info(f"Added global memory for user {user_id}")
        except Exception as e:
            logger.error(f"Error storing conversation embedding: {e}")

    async def enhanced_link_conversation(self, user_id: str, conversation_id: str, query: str) -> str:
        try:
            query_embedding = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)
            conversations = get_user_conversations(user_id)
            if not conversations:
                return conversation_id

            best_match, best_score = conversation_id, 0.0
            for conv in conversations:
                if conv["id"] == conversation_id:
                    continue
                short_term = get_short_term_history(conv["id"], limit=5)
                long_term = await self.global_memory.search_user_memory(user_id, query, top_k=3)
                conv_text = " ".join([msg["content"] for msg in short_term] + 
                                [doc.page_content for doc in long_term if doc.metadata.get("context") == "text"] + 
                                [doc.page_content for doc in long_term if doc.metadata.get("context") == "image_analysis"])
                if not conv_text:
                    continue
                conv_embedding = np.array(self.embedding_model.embed_query(conv_text)).reshape(1, -1)
                score = cosine_similarity(query_embedding, conv_embedding)[0, 0]
                if score > 0.85 and score > best_score:
                    best_match, best_score = conv["id"], score
            logger.debug(f"Best conversation match for query '{query}': {best_match} (score: {best_score})")
            return best_match
        except Exception as e:
            logger.error(f"Error linking conversation: {e}")
            return conversation_id
        
    async def query(self, question: str, k: int = 5, user_id: Optional[str] = None, conversation_id: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        logger.info(f"Processing query: {question}")
        if not user_id or not conversation_id:
            logger.warning("No user_id or conversation_id provided, history may be incomplete")

        try:
            history_str = await self.get_relevant_history(user_id, conversation_id, question, history=history)
            logger.debug(f"History for query:\n{history_str}")

            intent = self.detect_intent(question, history)
            logger.info(f"Detected intent: {intent}")

            if max(intent.values()) == intent.get("out_of_domain", 0):
                return {
                    "query": question,
                    "response": "I cannot answer questions about programming, personal security, specific areas, or locations. Please ask about road design, rules, or regulations.",
                    "status": "out_of_domain",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "images": []
                }

            # Check for image request intent
            is_image_request = intent.get("image_request", 0) > 0.5 or "show me" in question.lower()
            logger.debug(f"Is image request: {is_image_request} (Intent Score: {intent.get('image_request', 0)})")

            retrieved_docs = []
            image_refs = []

            # Text retrieval for context
            retrieved_docs = await self.text_vector_store.search_hybrid(question, k=k, cross_encoder=self.cross_encoder)
            context_str = ""
            for doc in retrieved_docs:
                context_str += f"{doc.page_content} (Source: {doc.metadata['file_name']}, Page {doc.metadata['page_number']})\n\n---\n\n"
                json_path = os.path.join(self.json_output_dir, f"{os.path.splitext(doc.metadata['file_name'])[0]}.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                    for img in doc_data.get("image_descriptions", []):
                        if img["page"] == doc.metadata["page_number"] - 1 and "description_json" in img:
                            validation = await self.validate_image(question, img["description_json"]["description"])
                            logger.debug(f"Text context image validation for {img['extracted_path']}: {validation}")
                            if validation["is_valid"]:
                                image_refs.append({
                                    "page": img["page"] + 1,
                                    "description": img["description_json"]["description"],
                                    "base64": img["base64"],
                                    "format": img["format"]
                                })

            if not retrieved_docs:
                context_str = "No relevant documents found. Providing a response based on available data."
                logger.debug("No text documents retrieved")

            # Fetch images explicitly
            if is_image_request:
                logger.info("Image request detected, searching for relevant images")
                additional_images = await self.image_metadata_manager.search_images(question, k=2, cross_encoder=self.cross_encoder, threshold=0.05)
                logger.debug(f"Raw search found {len(additional_images)} images")
                
                for img in additional_images:
                    logger.debug(f"Image candidate: {img['extracted_path']}, Description: {img['description_json']['description'][:100]}...")
                    validation = await self.validate_image(question, img["description_json"]["description"])
                    logger.debug(f"Image validation for {img['extracted_path']}: {validation}")
                    if validation["is_valid"]:
                        if not any(ref["base64"] == img["base64"] for ref in image_refs):  # Avoid duplicates
                            image_refs.append({
                                "page": img["page"],
                                "description": img["description_json"]["description"],
                                "base64": img["base64"],
                                "format": img["format"]
                            })

                logger.debug(f"Total images after validation: {len(image_refs)}")
                if not image_refs:
                    logger.warning("No relevant images found despite lenient search")

            # Prepare image descriptions for the prompt
            image_descriptions = ""
            if image_refs:
                for i, img in enumerate(image_refs, 1):
                    image_descriptions += f"\nImage {i} (Page {img['page']}): {img['description']}\n"
                logger.debug(f"Image descriptions for prompt:\n{image_descriptions}")

            # Prepare prompt
            prompt = self.rag_prompt_template.format(history=history_str, context=context_str, question=question)
            if image_refs:
                prompt += f"\n\nThe following images are available to reference in your response:\n{image_descriptions}\nIncorporate these images into your answer by referencing them (e.g., 'See Image 1 on Page X'). Do not say you cannot display images, as the images are attached. Provide a cohesive response that describes the image in the context of the query."
            else:
                prompt += "\n\nNo images are available. Provide a detailed text description instead."

            response = await asyncio.to_thread(self.llm.invoke, prompt)

            # Process response
            processed_response = self.post_process_response(response.content)
            # Post-process to remove any apology about not displaying images if images are attached
            if image_refs:
                processed_response = processed_response.replace(
                    "I apologize, but I’m a text-based AI assistant, and I don’t have the capability to display images.", ""
                ).replace(
                    "I apologize, but as a text-based AI assistant, I don't have the capability to display images.", ""
                ).strip()
                processed_response += f"\n\nAttached {len(image_refs)} relevant image(s) below for reference."
            elif is_image_request:
                processed_response += "\n\nNo relevant images found, but I’ve provided a detailed text description above."
            final_response = self.add_follow_up_suggestions(processed_response, question)

            logger.info(f"Response generated: {final_response[:200]}... (Images: {len(image_refs)})")

            return {
                "query": question,
                "response": final_response,
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "images": image_refs
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": question,
                "response": "An error occurred while processing your request. Please try again.",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "images": []
            }
    
    async def cleanup_index(self, max_age_days: int = 90):
        current_time = time.time()
        docs_to_keep = [doc for doc in self.text_vector_store.vector_store.docstore._dict.values() if (current_time - doc.metadata.get("timestamp", 0)) < max_age_days * 86400]
        self.text_vector_store.vector_store = FAISS.from_documents(docs_to_keep, self.text_vector_store.embedding_model)
        await asyncio.to_thread(self.text_vector_store.vector_store.save_local, self.text_vector_store.vector_db_path)

async def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not groq_api_key or not gemini_api_key:
        logger.error("API keys not found in environment variables")
        exit(1)

    rag_manager = MultiModalRAGManager(groq_api_key=groq_api_key, gemini_api_key=gemini_api_key, pdf_folder_path="./pdfs")
    logger.info("Starting document ingestion...")
    stats = await rag_manager.ingest_documents(force_reprocess=False)
    logger.info(f"Ingestion stats: {json.dumps(stats, indent=2)}")

    question = "What are the width guidelines for cycle lanes near pedestrian crossings?"
    result = await rag_manager.query(question, k=3)
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['response']}")

if __name__ == "__main__":
    asyncio.run(main())