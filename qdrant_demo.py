import os
import random
from qdrant_client.http.models import CollectionStatus, UpdateStatus, MatchValue, InitFrom
from qdrant_client.http.models import Distance, VectorParams, PointStruct, ReplicateShardOperation, \
    OptimizersConfigDiff, Filter, FieldCondition, PointIdsList, Batch
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import AsyncApis


class QdrantClientOperation:
    """
    Qdrant客户端, 用于操作Qdrant集群, 以及集群中的集合, 点, 以及搜索, 推荐等操作
    """

    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )
        self.model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

    def search(self, collection_name, query_vector, query_filter=None, limit=10, with_payload=True, with_vectors=False, score_threshold=None, offset=0):
        """
        在集合中搜索最接近的向量，考虑过滤条件，返回最相似的搜索结果列表。
        :param collection_name:
        :param query_vector:
        :param query_filter:
        :param limit:
        :param with_payload:
        :param with_vectors:
        :param score_threshold:
        :param offset:
        :return:
        example:
            query_filter=Filter(
                                 must=[
                                        FieldCondition(key="name", match=MatchValue(value="jeffrey"))
                                    ],
                                 ),
        """
        return self.qdrant_client.search(collection_name, query_vector, query_filter, limit=limit, with_payload=with_payload, with_vectors=with_vectors, score_threshold=score_threshold, offset=offset)

    def recommend(self, collection_name, positive, negative=None, limit=10):
        """
        推荐点：基于已存储在Qdrant中的示例搜索相似点。
        提供存储点的ID，Qdrant将基于已存在的向量执行搜索。
        此功能对于现有点集合的推荐特别有用。
        返回的结果是按照相似度排序的点列表。
        :param collection_name:
        :param positive:
        :param negative:
        :param limit:
        :return: list
        """
        return self.qdrant_client.recommend(collection_name, positive, negative, limit=limit)

    def scroll(self, collection_name, consistency=None, limit=10, with_payload=True, with_vectors=True):
        """
        滚动集合中的所有（匹配）点。
        此方法提供了一种方法来迭代所有存储的点，具有一些可选的过滤条件。
        滚动不适用任何相似度估计，它将按升序返回点的id。
        返回的结果是一个元组，元组中包含两个元素，第一个元素是点的列表，第二个元素是下一页的偏移量。
        如果下一页的偏移量是None，那么就没有更多的点了。
        :param collection_name:
        :param consistency:
        :param limit:
        :param with_payload:
        :param with_vectors:
        :return: tuple
        """
        return self.qdrant_client.scroll(collection_name, consistency, limit=limit, with_payload=with_payload, with_vectors=with_vectors)

    def count(self, collection_name, count_filter=None):
        """
        计算集合中的点数。
        计算与给定过滤器匹配的集合中的点数。
        返回符合过滤器的集合中的点数。
        :param collection_name:
        :param count_filter:
        :return:
        """
        return self.qdrant_client.count(collection_name, count_filter)

    def upsert(self, collection_name, points):
        """
        在集合中更新或插入新点。如果给定 ID 的点已经存在 -将被覆盖。
        返回操作结果。
        :param collection_name: 
        :param points: 
        :return: 
        """
        operation_info = self.qdrant_client.upsert(collection_name, points)
        assert operation_info.status == UpdateStatus.COMPLETED
        return operation_info

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=True):
        """
        从通过ID值检索点的数据。
        返回点列表。
        :param collection_name: 
        :param ids:
        :param with_payload:
        :param with_vectors:
        :return:
        """
        return self.qdrant_client.retrieve(collection_name, ids, with_payload, with_vectors)

    def delete(self, collection_name, points_selector):
        """
        从集合中删除选择的点。
        返回操作结果。
        :param collection_name: 
        :param points_selector: 
        :return: 
        """
        return self.qdrant_client.delete(collection_name, points_selector)

    def get_collections(self):
        """
        返回所有集合的列表。
        :return:
        """
        return self.qdrant_client.get_collections()

    def get_collection(self, collection_name):
        """
        获取有关指定现有集合的详细信息。
        :param collection_name:
        :return:
        """
        collection_info = self.qdrant_client.get_collection(collection_name)
        assert collection_info.status == CollectionStatus.GREEN
        return collection_info

    def update_collection(self, collection_name, optimizer_config=None, collection_params=None):
        """
        更新集合的元数据。
        :param collection_name:
        :param optimizer_config:
        :param collection_params:
        :return:
        """
        return self.qdrant_client.update_collection(collection_name,optimizer_config,collection_params)

    def delete_collection(self, collection_name):
        """
        删除集合及其包含的所有数据
        :param collection_name:
        :return:
        """
        return self.qdrant_client.delete_collection(collection_name)

    def create_collection(self, collection_name, vectors_config, on_disk_payload=False, shard_number=1,
                          quantization_config=None, init_from=None
                          ):
        """
        通过给定的参数创建新集合。
        返回操作结果。
        :param collection_name:
        :param vectors_config:
        :param on_disk_payload:
        :param shard_number:
        :param quantization_config:
        :param init_from:
        :return:
        example:
            collection_name="test_document_chunks",
            vectors_config=VectorParams(size=1024, distance=Distance.DOT), # Distance.DOT,Distance.COSINE,Distance.Euclid
            on_disk_payload=True
            shard_number=1
        """
        return self.qdrant_client.create_collection(collection_name, vectors_config, on_disk_payload, shard_number,
                                                    quantization_config, init_from)

    def recreate_collection(self, collection_name, vectors_config, shard_number=1, on_disk_payload=False, init_from=None, quantization_config=None):
        """
        通过给定的参数重新创建集合。会删除之前存在的集合，并通过给定的参数创建新集合
        返回操作结果。
        :param collection_name:
        :param vectors_config:
        :param shard_number:
        :param on_disk_payload:
        :param init_from:
        :param quantization_config:
        :return:
        example:
            collection_name="test_document_chunks",
            vectors_config=VectorParams(size=1024, distance=Distance.DOT),
            on_disk_payload=True
        """
        if collection_name not in self.get_collections():
            assert f"collection: {collection_name} is not exist!"
        self.delete_collection(collection_name)
        operation_result = self.create_collection(collection_name,vectors_config,on_disk_payload=on_disk_payload, shard_number=shard_number,quantization_config=quantization_config,init_from=init_from)
        return operation_result

    def upload_records(self, collection_name, records, parallel=1):
        """
        上传记录到集合
        类似于upload_collection方法，但是操作的是记录，而不是独立的向量和有效负载。
        这种方法更有效，因为它使用批量操作和并行性。
        返回操作结果。
        :param collection_name:
        :param records:
        :param parallel:
        :return:
        """
        return self.qdrant_client.upload_records(collection_name, records, parallel=parallel)

    def upload_collection(self, collection_name, vectors, payload=None, parallel=1):
        """
        上传向量和有效负载到集合。
        此方法将执行数据的自动批处理。
        如果需要执行单个更新，请使用`upsert`方法。
        注意：如果要上传多个向量和单个有效负载，请使用`upload_records`方法。
        返回操作结果。
        :param collection_name:
        :param vectors:
        :param payload:
        :param parallel:
        :return:
        """
        return self.qdrant_client.upload_collection(collection_name, vectors, payload=payload, parallel=parallel)

    def set_payload(self, collection_name, payload, points):
        """
        为点设置有效负载。
        返回操作结果。
        :param collection_name:
        :param payload:
        :param points: ids
        :return:
        """
        return self.qdrant_client.set_payload(collection_name, payload, points)

    def overwrite_payload(self, collection_name, payload, points):
        """
        对指定的点设置有效负载。
        在应用此操作后，只有指定的有效负载才会出现在点中。
        即使在有效负载中没有指定密钥，现有的有效负载也将被删除。
        返回操作结果。
        为点设置有效负载。
        :param collection_name:
        :param payload:
        :param points: ids
        :return:
        """
        return self.qdrant_client.overwrite_payload(collection_name, payload, points)

    def delete_payload(self, collection_name, keys, points):
        """
        从点的负载中删除值。
        返回操作结果。
        :param collection_name:
        :param keys:
        :param points: ids
        :return:
        """
        return self.qdrant_client.delete_payload(collection_name, keys, points)

    def clear_payload(self, collection_name, points_selector):
        """
        清除选中点的有效负载。
        返回操作结果。
        :param collection_name:
        :param points_selector:
        :return:
        """
        return self.qdrant_client.clear_payload(collection_name, points_selector)

    def create_payload_index(self, collection_name, field_name, field_schema):
        """
        创建一个有效负载字段的索引。
        索引字段允许更快地执行过滤的搜索操作。
        返回操作结果。
        :param collection_name:
        :param field_name:
        :param field_schema:
        :return:
        """
        return self.qdrant_client.create_payload_index(collection_name, field_name, field_schema)

    def delete_payload_index(self, collection_name, field_name):
        """
        删除有效负载字段的索引。
        返回操作结果。
        :param collection_name:
        :param field_name:
        :return:
        """
        return self.qdrant_client.delete_payload_index(collection_name, field_name)

    def list_snapshots(self, collection_name):
        """
        返回给定集合的所有快照列表。
        :param collection_name:
        :return:
        """
        return self.qdrant_client.list_snapshots(collection_name)

    def create_snapshot(self, collection_name):
        """
        为给定集合创建新快照。
        返回操作结果。
        :param collection_name:
        :return:
        """
        return self.qdrant_client.create_snapshot(collection_name)

    def delete_snapshot(self, collection_name, snapshot_name):
        """
        删除给定集合的指定快照。
        返回操作结果，True or False。
        :param collection_name:
        :param snapshot_name:
        :return:
        """
        return self.qdrant_client.delete_snapshot(collection_name, snapshot_name)

    def list_full_snapshots(self):
        """
        返回所有集合的所有快照列表。
        :return:
        """
        return self.qdrant_client.list_full_snapshots()

    def create_full_snapshot(self):
        """
        为所有集合创建新快照。
        返回快照列表。
        :return:
        """
        return self.qdrant_client.create_full_snapshot()

    def delete_full_snapshot(self, snapshot_name):
        """
        删除所有集合的指定快照。
        返回操作结果，True or False。
        :param snapshot_name:
        :return:
        """
        return self.qdrant_client.delete_full_snapshot(snapshot_name)

    def recover_snapshot(self, collection_name, location):
        """
        从快照中恢复集合。
        返回操作结果。
        :param collection_name:
        :param location:
        :return:
        """
        return self.qdrant_client.recover_snapshot(collection_name, location)


if __name__ == '__main__':
    from qdrant_client.http.models import Distance, VectorParams, PointStruct, ReplicateShardOperation, OptimizersConfigDiff, Filter, FieldCondition, PointIdsList, Batch
    import numpy as np
    collection = "news_document_chunks"
    qdrant_client = QdrantClientOperation()
    result = qdrant_client.get_collections()
    print(result)
    result = qdrant_client.get_collection(collection)
    print(result)

    query = [0.08, 0.86, 0.77, 0.91, 0.07, 0.89, 0.68, 0.4, 0.03, 0.86]
    search_result = qdrant_client.search(
        collection_name=collection,
        query_vector=("image", query),
        with_vectors=True,
        score_threshold=4.2
    )
    print(search_result)



