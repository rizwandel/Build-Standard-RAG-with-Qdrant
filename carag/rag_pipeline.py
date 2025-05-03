""" Cache Aware Retrieval Augmented Generation System """
# Author: Mohamed Rizwan <rizdelhi@gmail.com>

# Import required libraries
import time
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union, Dict
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, SearchParams, Distance, VectorParams
from qdrant_client.http.models.models import ScoredPoint
from fastembed import TextEmbedding, LateInteractionTextEmbedding
from fastembed.sparse.bm25 import Bm25
import logging
logging.getLogger("fastembed").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# parent class
class rag_pipe:
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str = None,
        dense_model_name: Optional[str] = "jinaai/jina-embeddings-v2-base-en",
        threshold: float = 0.5,
        sparse_model_name: Optional[str] = "Qdrant/bm25",
        late_interaction_model_name: Optional[str] = "jinaai/jina-colbert-v2",
        llm_model_name: Optional[str]="mistral-large-latest",
        mistral_api_key: Optional[str] = None,
        **kwargs
        ):
        """
        rag_pipe is a class that provides functionality for storing and searching embeddings in a Qdrant vector database.
        It supports both dense, sparse and late interaction embeddings, as well as a late interaction model for improved retrieval performance.
        We are using fastembed for dense and late interaction embeddings, and BM25 for sparse embeddings.
        It is recommended to use the latest version of Qdrant for best performance.
        The class also provides functionality for caching the results of queries to improve performance.
        The cache is implemented using Qdrant's in-memory cache, which is created on the fly.
        Note: The class is designed to be used with Qdrant cloud. 
        Not yet supports embedding models from sentence-transformers or hugging face.
        
        Args:
            url (str): The URL of the Qdrant server.
            api_key (str): The API key for the Qdrant server.
            collection_name (str): The name of the Qdrant collection to use (recommended to use lower case). Example: "european_ai_act_2024"
            dense_model_name (str): Defaults to "jinaai/jina-embeddings-v2-base-en".
            threshold (float): The threshold for the similarity score to consider a document relevant. Defaults to 0.7.
            cache_collection_name ([str]): The name of cache collection is "cache".
            sparse_model_name (Optional[str]): The name of the sparse model to use. Defaults to "Qdrant/bm25".
            late_interaction_model_name (Optional[str]): The name of the late interaction model to use. Defaults to "jinaai/jina-colbert-v2".
            llm_model_name (str): The name of the Mistral LLM, defaults to "mistral-large-latest"
            mistral_api_key (str): The MISTRAL_API_KEY
        """
        # initialize embedding models and sizes
        # Store model names as instance attributes
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.late_interaction_model_name = late_interaction_model_name 
        try:
            self.dense_model = TextEmbedding(dense_model_name) if dense_model_name else TextEmbedding("jinaai/jina-embeddings-v2-base-en")
            self.late_interaction_model = LateInteractionTextEmbedding(late_interaction_model_name) if late_interaction_model_name else None 
            self.sparse_model = Bm25(sparse_model_name) if sparse_model_name else None 
            self.dense_embedding_size = self._get_dense_embedding_size() 
            self.sparse_embedding_size = self._get_sparse_embedding_size() if self.sparse_model else None
            self.late_embedding_size = (
                self._get_late_embedding_size() 
                if self.late_interaction_model 
                else None
            )
        except Exception as e:
            logger.info(f"Failed to initialize embedding models: {e}")
            raise RuntimeError(f"Failed to initialize embedding models: {e}") from e
        # create cache client
        #self.cache_client = QdrantClient(path="./qdrant_cache", prefer_grpc=True) # need configure for concurrent access
        self.cache_client = QdrantClient(":memory:") # in-memory cache
        self.cache_collection_name = "cache" # cache collection name
        self.cache_distance_metric = Distance.COSINE if self.dense_model else Distance.EUCLIDEAN
        self.threshold = threshold
        self._initialize_cache_collection()
        # create Qdrant cloud client
        if not url or not api_key:
            raise ValueError("URL and API key must be provided.")
        if not collection_name:
            raise ValueError("Collection name must be provided.")
        self.url = url
        self.api_key = api_key
        self.db_client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.llm_model_name = llm_model_name
        self.mistral_api_key = mistral_api_key
        #self._create_or_use_qdrant_collection(collection_name)    
    # create a new Qdrant collection if it doesn't exist or use the existing one
    def _create_or_use_qdrant_collection(self, collection_name:str):
        """Create new Qdrant collection."""
        if not collection_name:
            raise ValueError("Collection name must be provided.")
        # Check if the collection already exists
        collection_name = collection_name.strip().lower()
        existing_collections = [c.name.lower() for c in self.db_client.get_collections().collections]
        # create a new collection if it doesn't exist
        if collection_name not in existing_collections:
            #print(f"Creating new collection: {collection_name}")
            self.db_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=self.dense_embedding_size,
                        distance=models.Distance.COSINE,
                        datatype=models.Datatype.FLOAT16,
                    ),
                    "late_interaction_vector": models.VectorParams(
                        size=self.late_embedding_size,
                        distance=models.Distance.COSINE,
                        datatype=models.Datatype.FLOAT16,
                        multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    ) if self.late_interaction_model else None,
                },
                sparse_vectors_config={
                    "sparse_vector": models.SparseVectorParams(
                        index=models.SparseIndexParams())
                } if self.sparse_model else None,
            )
            #print(f"Collection {collection_name} created successfully.")
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info("Qdrant collection is being used {collection_name}")
            print(f"Using existing collection: {collection_name}")
        self.collection_name = collection_name
        
    def _upload_text_chunks(self, text_chunks: List[dict], collection_name: str, batch_size: int=1):
        """calls collection setup internally"""
        self._create_or_use_qdrant_collection(collection_name)
        points = []
        for chunk in tqdm(text_chunks, desc="Processing chunks"):
            point = self._create_point(chunk)
            points.append(point)

        self.db_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
        # Verify upload
        count = self.db_client.count(collection_name=collection_name).count
        print(f"Successfully uploaded {len(text_chunks)} chunks. Total in collection: {count}")
        logger.info(f"Uploaded {len(text_chunks)} chunks to {collection_name}")      
    @classmethod        
    def upload_text_chunks(
        cls,
        url:str,
        api_key:str,
        text_chunks: List[dict],
        collection_name: str,
        batch_size: int = 1,
        **init_kwargs
        ):
        """Upload processed text chunks to the named collection with hybrid embeddings
        Args:
            text_chunks (List[dict]): List of dictionaries containing text chunks and metadata.
            collection_name (str): The name of the Qdrant collection to use. Defaults to "test".
            batch_size (int): The number of points to upload in each batch. Defaults to 1.
        """
        if not collection_name:
            raise ValueError("Collection name must be provided.")
        
        """Class method to upload chunks without instantiating the class"""
        instance = cls(url=url, api_key=api_key, collection_name=collection_name, **init_kwargs)
        instance._upload_text_chunks(text_chunks, collection_name, batch_size)
        
    def _create_point(self, chunk: dict) -> models.PointStruct:
        dense_embedding = self._get_dense_embedding(chunk['text'])
        sparse_embedding = self._get_sparse_embedding(chunk['text']) if self.sparse_model else None
        colbert_embedding = self._get_late_interaction_embedding(chunk['text']) if self.late_interaction_model else None    
        # Build vector payload
        vector_payload = {
            "dense_vector": dense_embedding,
            "late_interaction_vector": colbert_embedding if colbert_embedding is not None else None
        }
        # Add sparse vector if available
        if sparse_embedding:
            vector_payload["sparse_vector"] = models.SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
        return models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, chunk['text'])),  # Using incremental ID from enumeration
            vector=vector_payload,
            payload={
                "response": chunk['text'],
                "meta_data": chunk.get('metadata', {})
            }
        ) 
    
    # Dynamically determine embedding sizes
    def _get_dense_embedding_size(self) -> int:
        """Dynamically determine embedding size from the dense model"""
        try:
            # Get embedding size by encoding a test sample
            test_embedding = next(iter(self.dense_model.embed(["sample text"])))
            return len(test_embedding)
        except Exception as e:
            logger.error("Failed to determine embedding size")
            raise RuntimeError("Embedding size detection failed") from e 
    def _get_sparse_embedding_size(self) -> int:
        """Sparse embeddings are usually vectors with dynamic length, so we default or set a fake fixed size"""
        # BM25 sparse vectors are not dense vectors; no real "size", depends on number of words
        return None  # or return a placeholder if Qdrant needs something 
        
    def _get_late_embedding_size(self) -> int:
        """Return embedding size based on the stored model name."""
        try:
            model_name = self.late_interaction_model_name.lower()
            if model_name in ["jinaai/jina-colbert-v2", "colbert-ir/colbertv2.0","vidore/colpali-v1.3","vidore/colqwen2.5-v0.2"]:
                return 128
            elif model_name=="answerdotai/answerai-colbert-small-v1":
                return 96
            else:
                raise ValueError(f"Unsupported late interaction model: {model_name}")
        except Exception as e:
            logger.info(f"Failed to determine late embedding size: {e}")
            raise RuntimeError("Late embedding size detection failed.") from e
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

    def _get_embeddings_parallel(self, query: str):
        """Get all embeddings in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor() as executor:
            futures = {
                'dense': executor.submit(self._get_dense_embedding, query)
            }
            if self.sparse_model:
                futures['sparse'] = executor.submit(self._get_sparse_embedding, query)
            if self.late_interaction_model:
                futures['colbert'] = executor.submit(self._get_late_interaction_embedding, query)
            
            dense_vec = futures['dense'].result()
            sparse_vec = futures['sparse'].result() if 'sparse' in futures else None
            colbert_vec = futures['colbert'].result() if 'colbert' in futures else None
            
            return dense_vec, sparse_vec, colbert_vec

    def _initialize_cache_collection(self):
        """Initialize the cache collection in Qdrant: This method initializes the cache collection with the specified name and configuration.
        It also configures optimizers with a memory mapping threshold.
        Args:
            cache_collection_name (str): The name of the cache collection.Defaults to "cache".
        """
        collections = {c.name for c in self.cache_client.get_collections().collections}
        if self.cache_collection_name not in collections:
            self.cache_client.create_collection(
                collection_name=self.cache_collection_name,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=self.dense_embedding_size,
                        distance=self.cache_distance_metric,
                        datatype=models.Datatype.FLOAT16,
                        ),
                },
                optimizers_config=models.OptimizersConfigDiff(memmap_threshold=10000),
            )
        logger.info(f"Cache collection created successfully: {self.cache_collection_name}")
    def add_to_cache(self, query:str, response:str, score: float):
        try:
            point = models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, query)),
                vector={"dense_vector":list(self.dense_model.embed([query]))[0].tolist()},
                payload={
                    "query": query,
                    "response": response,
                    "score": score,
                    "meta_data": {},
                    "timestamp": time.time(),
                }
            )
            # upload the point to the cache
            self.cache_client.upsert(
                collection_name=self.cache_collection_name,
                points=[point],
                wait=True,
            )
            #logger.info(f"Added to cache: {query} -> {response}") 
            #print(f"Added to cache: {query} -> {response}")  
        except Exception as e:
            #logger.info(f"Failed to add to cache: {e}")
            # print(f"Failed to add to cache: {e}")
            raise RuntimeError(f"Failed to add to cache: {e}") from e 

    def search_cache(
        self, 
        query: str, 
        threshold: float = 0.7
        ) -> Optional[models.ScoredPoint]:
        """ 
        Search the cache for a given query.
        Args:
            query (str): The query to search for.
            threshold (float): The threshold for the similarity score to consider a document relevant. Defaults to 0.7.
        """
        if not self.cache_collection_name:
            raise ValueError("Cache collection not exists.")
        self.cache_collection_name = self.cache_collection_name.strip()
        
        query_embedding = list(self.dense_model.embed([query]))[0].tolist()
        results = self.cache_client.search(
            collection_name=self.cache_collection_name,
            query_vector =("dense_vector",query_embedding),
            limit=1,
            score_threshold=self._get_cache_threshold(threshold),
        )
        if not results:
            return None
                
        cached_result = results[0]
        semantic_score = cached_result.score
        original_score = cached_result.payload.get("score", 0.0)
        combined_score = (semantic_score + original_score) / 2
        if combined_score > self._get_cache_threshold(threshold):
            logger.info(f"Cache hit: {query} -> {cached_result.payload['response']}")
            return {
                "id": cached_result.id,
                "response": cached_result.payload["response"],
                "semantic_score": cached_result.score,
                "original_score": cached_result.payload.get("score", 0.0),
                "combined_score": combined_score
            }
        else:
            logger.info(f"Cache miss: {query} -> No relevant result found.")
            return None
            
    def _get_cache_threshold(self, threshold: float=None) -> float:
        if threshold is not None:
            return threshold
        return 0.7 if self.cache_distance_metric == Distance.COSINE else 0.1
    def _format_result(self, result, elapsed_time: float) -> dict:
        """Format the cache hit result with metadata."""
        try:
            if not result:
                return {
                    "response": None,
                    "cache_hit": False,
                    "elapsed_time": elapsed_time
                }
            # Extract response and scores from the cached result
            return {
                "id": result['id'],
                "response": result['response'],
                "cache_hit": True,
                "semantic_score": result['semantic_score'],
                "original_score": result['original_score'],
                "elapsed_time": elapsed_time
            }

        except KeyError as e:
            logger.error(f"Malformed cache result payload: {str(e)}")
            raise ValueError(f"Cache result missing expected key: {str(e)}") from e
     
    def retrieve(
        self,
        query: str,
        collection_name: str,
        cache_first: bool = True,
        top_k: Optional[int]=20) -> List[models.ScoredPoint]:
        """
        Retrieves relevant documents for a given query from the Qdrant vector database, optionally using cache for faster results.
        This method first checks the cache for a matching response and, if not found, performs a hybrid search using dense, sparse, and late interaction embeddings.

        Args:
            query (str): The query string to search for.
            collection_name (str): The name of the Qdrant collection to search.
            cache_first (bool): Whether to check the cache before querying the database. Defaults to True.
            top_k (Optional[int]): The number of top results to return. Defaults to 20.

        Returns:
            List[models.ScoredPoint]: A list of scored points representing the most relevant documents.
        """
        start_time = time.time()
        # Cache check
        if cache_first:
            if cached := self.search_cache(query):
                logger.info(f"Cache hit for query: {query}")
                return self._format_result(cached, time.time()-start_time)
        
        # Get embeddings in parallel
        dense_vec, sparse_vec, colbert_vec = self._get_embeddings_parallel(query)
        # Prepare sparse vector if it exists
        #sparse_vector = None
        if sparse_vec:
                sparse_vectors = models.SparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values
                )
        # Build hybrid query 
        try:
            results = self.db_client.query_points(
                collection_name=self.collection_name,
                prefetch=models.Prefetch(
                    query=sparse_vectors,
                    using="sparse_vector",
                    limit=1000
            ),
            query=dense_vec,
            using="dense_vector",
            limit=100,
            with_payload=True
            ),
            query=colbert_vec,
            using="late_interaction_vector",
            limit=100,
            with_payload=True
        except Exception as e:
            logger.error(f"Error during Qdrant query: {e}")
            raise RuntimeError(f"Query failed: {e}") from e
        # Check if results are empty
        if not results:
            logger.warning("No results found in the database.")
            return []
        # update cache 
        if results[0]:
            query_response = results[0]
            response = query_response.points[0].payload["response"]
            score = query_response.points[0].score
            self.add_to_cache(str(query),str(response), int(score))
        query_responses = results[0].points
        
        return query_responses[:top_k]  # Return top K results
    
    # invoke method to handle the query and return the response - This method will first check the cache and then search the database if not found in the cache 
    @classmethod
    def invoke(
        cls,
        url: str,
        api_key: str,
        query: str,
        collection_name: str,
        **init_kwargs
    ) -> Union[str, List[models.ScoredPoint]]:
        instance = cls(
            url=url,
            api_key=api_key,
            collection_name=collection_name,
            **init_kwargs
        )
        return instance.retrieve(query=query, collection_name=collection_name)
    
    


    

                    
