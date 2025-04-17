""" Cache Aware Retrieval Augmented Generation System """
# Author: Mohamed Rizwan <rizdelhi@gmail.com>

# Import required libraries
import time
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.models import SearchParams, Distance
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, SearchParams,Distance
from qdrant_client.http.models import SearchParams
from fastembed import TextEmbedding, LateInteractionTextEmbedding
from fastembed.sparse.bm25 import Bm25
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# parent class
class rag_pipe:
    """
    Cache_RAG_pipeline (Cache aware Retrieval Augmented by Qdrant) is a class that provides functionality for storing and searching document embeddings in a Qdrant vector database.
    It supports both dense, sparse and late interaction embeddings, as well as a late interaction model for improved retrieval performance.
    Args:
        url (str): The URL of the Qdrant server.
        api_key (str): The API key for the Qdrant server.
        collection_name (str): The name of the Qdrant collection to use. Defaults to "European_AI_Act_2024_PDF_Multi"
        dense_model_name (str): Defaults to "jinaai/jina-embeddings-v2-base-en".
        threshold (float): The threshold for the similarity score to consider a document relevant. Defaults to 0.9.
        late_interaction_model_name (Optional[str]): The name of the late interaction model to use. Defaults to "jinaai/jina-colbert-v2".
        sparse_model_name (Optional[str]): The name of the sparse model to use. Defaults to "Qdrant/bm25".
        cache_collection_name (Optional[str]): The name of the default cache name
    """
    def __init__(self,
                 url: str,
                 api_key: str,
                 collection_name: str = "European_AI_Act_2024_PDF_Multi",
                 llm_model_name: Optional[str]="mistral-large-latest",
                 dense_model_name: str = "jinaai/jina-embeddings-v2-base-en",
                 threshold: float = 0.9,
                 late_interaction_model_name: Optional[str] = "jinaai/jina-colbert-v2",
                 sparse_model_name: Optional[str] = "Qdrant/bm25",
                 cache_collection_name: str = "qdrant_cache",
                 mistral_api_key: str = None
                 ):
        # initialize embedding models
        try:
            self.dense_model = TextEmbedding(dense_model_name)
            self.dense_embedding_size = self._get_dense_embedding_size() 
            self.late_interaction_model = None
            if late_interaction_model_name:
                self.late_interaction_model = LateInteractionTextEmbedding(
                    late_interaction_model_name)
            # Initialize the sparse model
            self.sparse_model = None
            if sparse_model_name:
                self.sparse_model = Bm25(sparse_model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")
            raise RuntimeError(f"Failed to initialize sparse embedding model: {e}") from e
         
        # create cache client
        #self.cache_client = QdrantClient(path="./qdrant_cache")
        self.cache_client = QdrantClient(":memory:")
        self.cache_collection_name = cache_collection_name
        self.euclidean_threshold = threshold
        self.distance_metric = Distance.COSINE

        try:
            self._initialize_cache_collection()
        except Exception as e:
            logger.error(f"Error during cache collection initialization: {e}")
            raise

         # Initialize Qdrant client
        self.db_client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.api_key = api_key
        self.url = url
        self._create_or_use_qdrant_collection()
    
    def upload_text_chunks(self, text_chunks: List[dict], batch_size: int = 1):
        """Upload processed text chunks to Qdrant collection with hybrid embeddings"""
        points = []
        for chunk in tqdm(text_chunks, desc="Processing text chunks"):
            dense_embedding = self._get_dense_embedding(chunk['text'])
            sparse_embedding = None
            if self.sparse_model:
                sparse_embedding = self._get_sparse_embedding(chunk['text'])  
            colbert_embedding = None
            if self.late_interaction_model:
                colbert_embedding = self._get_late_interaction_embedding(chunk['text'])
            # Build vector payload
            vector_payload = {
                "jina-embeddings-v2-base-en": dense_embedding,
                "jina-colbert-v2": colbert_embedding if self.late_interaction_model is not None else None
            }
            # Add sparse vector if available
            if sparse_embedding and self.sparse_model:
                vector_payload["bm25"] = models.SparseVector(
                    indices=sparse_embedding.indices.tolist(),
                    values=sparse_embedding.values.tolist()
                )
            # create a unique ID for the point
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk['text'])) 

            point = models.PointStruct(
                id=point_id,  # Using incremental ID from enumeration
                vector=vector_payload,
                payload={
                    "response": chunk['text'],
                    "meta_data": chunk.get('metadata', {})
                }
            )
            points.append(point)
            # Batch upload with error handling
            if len(points) >= batch_size:
                self.db_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                points = []  # Reset points after upload
        # Upload any remaining points
        if points:
            self.db_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        # Verify upload
        count = self.db_client.count(collection_name=self.collection_name).count
        print(f"Successfully uploaded {len(text_chunks)} chunks. Total in collection: {count}")

    def _get_dense_embedding_size(self) -> int:
        """Dynamically determine embedding size from the dense model"""
        try:
            # Get embedding size by encoding a test sample
            test_embedding = next(iter(self.dense_model.embed(["sample text"])))
            return len(test_embedding)
        except Exception as e:
            logger.error("Failed to determine embedding size")
            raise RuntimeError("Embedding size detection failed") from e  
        
    def _initialize_cache_collection(self):
        cache_collections = [collection.name for collection in self.cache_client.get_collections().collections]
        if self.cache_collection_name in cache_collections:
            logger.info(f"Using existing cache collection: {self.cache_collection_name}")
        else:
            self.cache_client.create_collection(
                collection_name=self.cache_collection_name,
                vectors_config={
                    "jina-embeddings-v2-base-en":models.VectorParams(
                        size=self._get_dense_embedding_size(),
                        distance=models.Distance.COSINE
                    ),
                },
                optimizers_config=models.OptimizersConfigDiff(memmap_threshold=10000),
            )
            #logger.info(f"Cache collection created successfully: {self.cache_collection_name}")
        # Check if the collection is empty
        collection_info = self.cache_client.get_collection(self.cache_collection_name)
        if collection_info != 0:
            logger.info(f"Cache collection not is empty: {self.cache_collection_name}")
      
    # create a new Qdrant collection if it doesn't exist or use the existing one
    def _create_or_use_qdrant_collection(self):
        existing_collections = [collection.name for collection in self.db_client.get_collections().collections]
        if self.collection_name not in existing_collections:
            self.db_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "jina-embeddings-v2-base-en": models.VectorParams(
                        size=self._get_dense_embedding_size(),
                        distance=models.Distance.COSINE,
                        datatype=models.Datatype.FLOAT16
                    ),
                    "jina-colbert-v2": models.VectorParams(
                        size=128, #hardcoded
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        index=models.SparseIndexParams(datatype=models.Datatype.FLOAT16))
                },
            )
            #logger.info(f"Created collection: {self.collection_name}")
        else:
            #logger.info("Qdrant collection is being used {self.collection_name}")
            logger.info(f"Existing Qdrant Collection Name is: {self.collection_name}")

    def _get_dense_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a given text using the dense model.
        Args:
            text (str): The text to embed.
        Returns:
            List[float]: The dense embedding vector.
        """
        # Get embedding - ensure we handle both single and batch cases
        if not isinstance(self.dense_model, TextEmbedding):
            return list(self.dense_model.embed(text))[0]
        embeddings = list(self.dense_model.embed([text]))  # Wrap in list
        return embeddings[0].tolist()  # Convert numpy array to list
    def _get_sparse_embedding(self, text: str):
        """
        Get the sparse embedding for a given text.
        Args:
            text (str): The text to embed.
        Returns:
            models.SparseVector: The sparse embedding vector.
        """
        if self.sparse_model is None:
            raise ValueError("Sparse model is not initialized.")
        return next(iter(self.sparse_model.embed(text)))
    def _get_late_interaction_embedding(self, text: str):
        """
        Get the late interaction embedding for a given text.
        Args:
            text (str): The text to embed.
        Returns:
            List[float]: The late interaction embedding vector.
        """
        if self.late_interaction_model is not None:
            return list(self.late_interaction_model.embed(text))[0]
        else:
            raise ValueError("Late interaction model is not initialized.")
    
    def search_cache(self, query: str):
        try:
            results= self.cache_client.search(
                collection_name=self.cache_collection_name,
                query_vector=(
                    "jina-embeddings-v2-base-en",
                    next(iter(self.dense_model.embed([query]))).tolist(),
                ),
                limit=1,
                score_threshold=self.euclidean_threshold,
                search_params=SearchParams(
                    unsw_ef=128,
                    exact=False
                    )
            ) 
            if results and self._is_relevant(results[0].score, self.distance_metric):
                return results[0]
            return None
        except Exception as e:
                logger.info(f"Cache search failed: {str(e)}")
        return None
    
    def _is_relevant(self, score: float, distance_metric: Distance) -> bool:
        if distance_metric == Distance.COSINE:
            return score <= self.euclidean_threshold  # Lower = better
        else:  
            return score >= self.euclidean_threshold  # Higher = better
    
    def add_to_cache(self, query:str, response:str):
        vector_name = "jina-embeddings-v2-base-en"
        # create a unique ID for the query and response text
        point_id = str(uuid.uuid4())
        point = PointStruct(
            id=point_id,
            vector={vector_name:list(self.dense_model.embed([query]))[0].tolist()},
            payload={
                "query": query,
                "response": response,
            }
        )
        # upload the point to the cache
        self.cache_client.upsert(
            collection_name=self.cache_collection_name,
            points=[point],
            wait=True,
        )
        logger.info(f"Added to cache: {query} -> {response}")    
    
    def search(self,query: str) -> List[models.ScoredPoint]:
        # Parallel embedding generation
        with ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
            dense_future = executor.submit(self._get_dense_embedding, query)
            sparse_future = executor.submit(self._get_sparse_embedding, query)
            colbert_future = executor.submit(self._get_late_interaction_embedding, query)
    
            # Wait for results from all futures
            dense,sparse,colbert = dense_future.result(),sparse_future.result(),colbert_future.result()

            # Prepare sparse vector if it exists
            sparse_vector = None
            if sparse:
                sparse_vector = models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist()
                )

            logger.debug(f"Type of colbert: {type(colbert)}, Value: {colbert}")
            # Build hybrid query 
            try:
                results = self.db_client.query_points(
                    collection_name=self.collection_name,
                    prefetch=models.Prefetch(
                        query=sparse_vector,
                        using="bm25",
                        limit=1000
                     ),
                     query=dense,
                     using="jina-embeddings-v2-base-en",
                     limit=100,
                     with_payload=True
                     ),
                query=colbert,#[colbert]
                using="jina-colbert-v2",
                limit=50,
                with_payload=True
            
            except Exception as e:
                logger.error(f"Error during Qdrant query: {e}")
                raise RuntimeError(f"Query failed: {e}") from e
            # Check if results are empty
            if not results:
                logger.warning("No results found in the database.")
                return []
            return results[0].points  # Returning ScoredPoint objects
    
    # invoke method to handle the query and return the response
    # This method will first check the cache and then search the database if not found in the cache
    def _format_result(self, result, elapsed_time):
        """Consistent result formatting"""
        #logger.info(f"result_id: {result.id}, Answer:\n{result.payload['response']}\nScore: {result.score}")
        #logger.info(f"Time taken: {elapsed_time:.2f}s")
        return result.payload['response']
    
    def invoke(self, query: str):
        start_time = time.time()
        if cached := self.search_cache(query):
            #logger.info(f"Cache hit")
            return self._format_result(cached, time.time()-start_time)
        
        db_results = self.search(query)
        if db_results and db_results[0]:
            #logger.info(f"DB hit: {query}")
            return db_results
        
        logger.info("No valid results found to add to cache.")
        return None
    
    def close(self):
        """Close the cache client"""
        self.cache_client.close()
        #print("cache client closed.")
        #logger.info("cache client closed.")
    
    


    

                    
