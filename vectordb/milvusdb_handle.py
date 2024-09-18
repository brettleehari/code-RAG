import logging
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus import model
from openai_embedding_function import OpenAIEmbeddingFunction

import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Connect to Milvus server
class MilvusDBHandle:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    model_name = 'text-embedding-3-large'
    dimensions = 512

    def __init__(self, host="localhost", port="19530"):
        logger.info(f"Connecting to Milvus at {host}:{port}")
        self.connection = connections.connect("default", host=host, port=port)

    def define_schema(self, fields):
        logger.debug(f"Defining schema with fields: {fields}")
        field_schemas = [FieldSchema(name=field['name'], dtype=field['dtype'], is_primary=field.get('is_primary', False)) for field in fields]
        schema = CollectionSchema(fields=field_schemas)
        logger.debug(f"Defined schema: {schema}")
        return schema

    def get_schema(self, collection_name):
        logger.debug(f"Getting schema for collection: {collection_name}")
        collection = Collection(name=collection_name)
        schema = collection.schema
        logger.debug(f"Schema for collection {collection_name}: {schema}")
        return schema
        
    def create_collection(self, collection_name, schema):
        logger.info(f"Creating collection: {collection_name} with schema: {schema}")
        collection = Collection(name=collection_name, schema=schema)
        logger.debug(f"Created collection: {collection}")
        return collection

    def drop_collection(self, collection_name):
        logger.info(f"Dropping collection: {collection_name}")
        collection = Collection(name=collection_name)
        collection.drop()
        logger.debug(f"Dropped collection: {collection_name}")

    def list_collections(self):
        logger.debug("Listing all collections")
        collections = connections.get_connection().list_collections()
        logger.debug(f"Collections: {collections}")
        return collections

    def collection_exists(self, collection_name):
        logger.debug(f"Checking if collection exists: {collection_name}")
        exists = collection_name in self.list_collections()
        logger.debug(f"Collection {collection_name} exists: {exists}")
        return exists

    def insert_vectors(self, collection_name, vectors, ids):
        logger.info(f"Inserting vectors into collection: {collection_name}")
        logger.debug(f"Vectors: {vectors}, IDs: {ids}")
        collection = Collection(name=collection_name)
        collection.insert([ids, vectors])
        logger.debug(f"Inserted vectors into collection: {collection_name}")

    def delete_vectors(self, collection_name, ids):
        logger.info(f"Deleting vectors from collection: {collection_name}")
        logger.debug(f"IDs: {ids}")
        collection = Collection(name=collection_name)
        expr = f"id in {ids}"
        collection.delete(expr)
        logger.debug(f"Deleted vectors from collection: {collection_name}")

    def create_index(self, collection_name, field_name, index_params):
        logger.info(f"Creating index on collection: {collection_name}, field: {field_name}")
        logger.debug(f"Index params: {index_params}")
        collection = Collection(name=collection_name)
        collection.create_index(field_name, index_params)
        logger.debug(f"Created index on collection: {collection_name}, field: {field_name}")

    def drop_index(self, collection_name, field_name):
        logger.info(f"Dropping index from collection: {collection_name}, field: {field_name}")
        collection = Collection(name=collection_name)
        collection.drop_index(field_name)
        logger.debug(f"Dropped index from collection: {collection_name}, field: {field_name}")

    def search_vectors(self, collection_name, query_vector, top_k, metric_type, params):
        logger.info(f"Searching vectors in collection: {collection_name}")
        logger.debug(f"Query vector: {query_vector}, top_k: {top_k}, metric_type: {metric_type}, params: {params}")
        collection = Collection(name=collection_name)
        search_params = {"metric_type": metric_type, **params}
        results = collection.search([query_vector], "vector_field", search_params, top_k)
        logger.debug(f"Search results: {results}")
        return results

    def hybrid_search(self, collection_name, query_vector, filters, top_k, metric_type, params):
        logger.info(f"Performing hybrid search in collection: {collection_name}")
        logger.debug(f"Query vector: {query_vector}, filters: {filters}, top_k: {top_k}, metric_type: {metric_type}, params: {params}")
        collection = Collection(name=collection_name)
        search_params = {"metric_type": metric_type, **params}
        results = collection.search([query_vector], "vector_field", search_params, top_k, expr=filters)
        logger.debug(f"Hybrid search results: {results}")
        return results
    
    def get_collection_stats(self, collection_name):    
        logger.info(f"Getting statistics for collection: {collection_name}")
        collection = Collection(name=collection_name)
        stats = collection.stats()
        logger.debug(f"Collection stats: {stats}")
        return stats
    
    def count_vectors(self, collection_name):
        logger.info(f"Counting vectors in collection: {collection_name}")
        collection = Collection(name=collection_name)
        count = collection.num_entities
        logger.debug(f"Number of vectors in collection: {count}")
        return count

    def create_openai_embedding_function(self):
        logger.info("Creating OpenAIEmbeddingFunction")
        openai_ef = OpenAIEmbeddingFunction(api_key = self.openai_api_key, model_name = self.model_name)    
        logger.debug("Created OpenAIEmbeddingFunction")
        return openai_ef
    
    def create_embeddings(self, documents):
        if not self.openai_ef:
            raise ValueError("OpenAIEmbeddingFunction is not initialized. Please provide an API key.")
        logger.info(f"Creating embeddings for documents: {documents}")
        embeddings = self.openai_ef.encode_documents(documents)
        logger.debug(f"Created embeddings: {embeddings}")
        return embeddings

    def insert_documents_with_embeddings(self, collection_name, documents, ids):
        logger.info(f"Inserting documents with embeddings into collection: {collection_name}")
        embeddings = self.create_embeddings(documents)
        self.insert_vectors(collection_name, embeddings, ids)
        logger.debug(f"Inserted documents with embeddings into collection: {collection_name}")



    def create_reranker(self, model_name="BAAI/bge-reranker-v2-m3", use_fp16=True, batch_size=32, normalize=True, device=None):
        logger.info(f"Creating BGERerankFunction with model_name: {model_name}, use_fp16: {use_fp16}, batch_size: {batch_size}, normalize: {normalize}, device: {device}")
        self.reranker = BGERerankFunction(model_name=model_name, use_fp16=use_fp16, batch_size=batch_size, normalize=normalize, device=device)
        logger.debug("Created BGERerankFunction")
        return self.reranker

    def rerank_results(self, query, results):
        if not hasattr(self, 'reranker'):
            raise ValueError("BGERerankFunction is not initialized. Please create a reranker first.")
        logger.info(f"Reranking results for query: {query}")
        reranked_results = self.reranker.rerank(query, results)
        logger.debug(f"Reranked results: {reranked_results}")
        return reranked_results
    
