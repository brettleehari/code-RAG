import unittest
from unittest.mock import patch, MagicMock
from pymilvus import CollectionSchema, FieldSchema, DataType
from code_RAG.vectordb.milvusdb_handle import MilvusDBHandle

class TestMilvusDBHandle(unittest.TestCase):

    @patch('code_RAG.vectordb.milvusdb_handle.connections.connect')
    def setUp(self, mock_connect):
        self.db_handle = MilvusDBHandle()
        mock_connect.assert_called_once_with("default", host="localhost", port="19530")

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_define_schema(self, mock_collection):
        fields = [{'name': 'field1', 'dtype': DataType.INT64, 'is_primary': True}]
        schema = self.db_handle.define_schema(fields)
        self.assertIsInstance(schema, CollectionSchema)
        self.assertEqual(len(schema.fields), 1)
        self.assertEqual(schema.fields[0].name, 'field1')

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_get_schema(self, mock_collection):
        mock_collection.return_value.schema = 'mock_schema'
        schema = self.db_handle.get_schema('test_collection')
        self.assertEqual(schema, 'mock_schema')
        mock_collection.assert_called_once_with(name='test_collection')

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_create_collection(self, mock_collection):
        schema = MagicMock()
        collection = self.db_handle.create_collection('test_collection', schema)
        mock_collection.assert_called_once_with(name='test_collection', schema=schema)
        self.assertEqual(collection, mock_collection.return_value)

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_drop_collection(self, mock_collection):
        collection = MagicMock()
        mock_collection.return_value = collection
        self.db_handle.drop_collection('test_collection')
        collection.drop.assert_called_once()

    @patch('code_RAG.vectordb.milvusdb_handle.connections.get_connection')
    def test_list_collections(self, mock_get_connection):
        mock_get_connection.return_value.list_collections.return_value = ['collection1', 'collection2']
        collections = self.db_handle.list_collections()
        self.assertEqual(collections, ['collection1', 'collection2'])

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_collection_exists(self, mock_collection):
        self.db_handle.list_collections = MagicMock(return_value=['collection1', 'collection2'])
        exists = self.db_handle.collection_exists('collection1')
        self.assertTrue(exists)

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_insert_vectors(self, mock_collection):
        collection = MagicMock()
        mock_collection.return_value = collection
        self.db_handle.insert_vectors('test_collection', [[1, 2, 3]], [1])
        collection.insert.assert_called_once_with([[1], [[1, 2, 3]]])

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_delete_vectors(self, mock_collection):
        collection = MagicMock()
        mock_collection.return_value = collection
        self.db_handle.delete_vectors('test_collection', [1])
        collection.delete.assert_called_once_with('id in [1]')

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_create_index(self, mock_collection):
        collection = MagicMock()
        mock_collection.return_value = collection
        self.db_handle.create_index('test_collection', 'field1', {'index_type': 'IVF_FLAT', 'params': {'nlist': 128}})
        collection.create_index.assert_called_once_with('field1', {'index_type': 'IVF_FLAT', 'params': {'nlist': 128}})

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_drop_index(self, mock_collection):
        collection = MagicMock()
        mock_collection.return_value = collection
        self.db_handle.drop_index('test_collection', 'field1')
        collection.drop_index.assert_called_once_with('field1')

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_search_vectors(self, mock_collection):
        collection = MagicMock()
        mock_collection.return_value = collection
        collection.search.return_value = 'search_results'
        results = self.db_handle.search_vectors('test_collection', [1, 2, 3], 10, 'L2', {'param': 'value'})
        collection.search.assert_called_once_with([[1, 2, 3]], 'vector_field', {'metric_type': 'L2', 'param': 'value'}, 10)
        self.assertEqual(results, 'search_results')

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_hybrid_search(self, mock_collection):
        collection = MagicMock()
        mock_collection.return_value = collection
        collection.search.return_value = 'hybrid_search_results'
        results = self.db_handle.hybrid_search('test_collection', [1, 2, 3], 'id > 0', 10, 'L2', {'param': 'value'})
        collection.search.assert_called_once_with([[1, 2, 3]], 'vector_field', {'metric_type': 'L2', 'param': 'value'}, 10, expr='id > 0')
        self.assertEqual(results, 'hybrid_search_results')

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_get_collection_stats(self, mock_collection):
        collection = MagicMock()
        mock_collection.return_value = collection
        collection.stats.return_value = 'stats'
        stats = self.db_handle.get_collection_stats('test_collection')
        self.assertEqual(stats, 'stats')

    @patch('code_RAG.vectordb.milvusdb_handle.Collection')
    def test_count_vectors(self, mock_collection):
        collection = MagicMock()
        mock_collection.return_value = collection
        collection.num_entities = 100
        count = self.db_handle.count_vectors('test_collection')
        self.assertEqual(count, 100)

    @patch('code_RAG.vectordb.milvusdb_handle.OpenAIEmbeddingFunction')
    def test_create_openai_embedding_function(self, mock_openai_embedding_function):
        self.db_handle.openai_api_key = 'test_key'
        openai_ef = self.db_handle.create_openai_embedding_function()
        mock_openai_embedding_function.assert_called_once_with(api_key='test_key', model_name='text-embedding-3-large')
        self.assertEqual(openai_ef, mock_openai_embedding_function.return_value)

    @patch('code_RAG.vectordb.milvusdb_handle.OpenAIEmbeddingFunction')
    def test_create_embeddings(self, mock_openai_embedding_function):
        self.db_handle.openai_ef = mock_openai_embedding_function.return_value
        mock_openai_embedding_function.return_value.encode_documents.return_value = 'embeddings'
        embeddings = self.db_handle.create_embeddings(['doc1', 'doc2'])
        self.assertEqual(embeddings, 'embeddings')

    @patch('code_RAG.vectordb.milvusdb_handle.MilvusDBHandle.insert_vectors')
    @patch('code_RAG.vectordb.milvusdb_handle.MilvusDBHandle.create_embeddings')
    def test_insert_documents_with_embeddings(self, mock_create_embeddings, mock_insert_vectors):
        mock_create_embeddings.return_value = 'embeddings'
        self.db_handle.insert_documents_with_embeddings('test_collection', ['doc1', 'doc2'], [1, 2])
        mock_create_embeddings.assert_called_once_with(['doc1', 'doc2'])
        mock_insert_vectors.assert_called_once_with('test_collection', 'embeddings', [1, 2])

    @patch('code_RAG.vectordb.milvusdb_handle.BGERerankFunction')
    def test_create_reranker(self, mock_bge_rerank_function):
        reranker = self.db_handle.create_reranker()
        mock_bge_rerank_function.assert_called_once_with(model_name="BAAI/bge-reranker-v2-m3", use_fp16=True, batch_size=32, normalize=True, device=None)
        self.assertEqual(reranker, mock_bge_rerank_function.return_value)

    @patch('code_RAG.vectordb.milvusdb_handle.BGERerankFunction')
    def test_rerank_results(self, mock_bge_rerank_function):
        self.db_handle.reranker = mock_bge_rerank_function.return_value
        mock_bge_rerank_function.return_value.rerank.return_value = 'reranked_results'
        results = self.db_handle.rerank_results('query', 'results')
        self.assertEqual(results, 'reranked_results')

if __name__ == '__main__':
    unittest.main()