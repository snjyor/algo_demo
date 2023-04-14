import os
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import AsyncApis


class QdrantClientDemo:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

    def get_qdrant_client(self):
        return self.qdrant_client

    def create_collection(self, collection_name):
        self.qdrant_client.create_collection(collection_name)

    def delete_collection(self, collection_name):
        self.qdrant_client.delete_collection(collection_name)

    def get_collection(self, collection_name):
        return self.qdrant_client.get_collection(collection_name)

    def get_collections(self):
        return self.qdrant_client.get_collections()
    
    def create_collection_if_not_exists(self, collection_name):
        if collection_name not in self.get_collections():
            self.create_collection(collection_name)
    
    def search_batch(self, collection_name, requests):
        return self.qdrant_client.search_batch(collection_name, requests)
    
    def search(self, collection_name, query_vector, limit=10):
        return self.qdrant_client.search(collection_name, query_vector, limit=limit)
    
    def recommend_batch(self, collection_name, requests):
        return self.qdrant_client.recommend_batch(collection_name, requests)
    
    def recommend(self, collection_name, positive, negative=None, limit=10):
        return self.qdrant_client.recommend(collection_name, positive, negative, limit=limit)
    
    def scroll(self, collection_name,consistency=None, limit=10):
        return self.qdrant_client.scroll(collection_name, consistency, limit=limit)
    
    def count(self, collection_name):
        return self.qdrant_client.count(collection_name)
    
    def upsert(self, collection_name, points):
        """
        在集合中更新或插入新点。如果给定 ID 的点已经存在 -将被覆盖。
        :param collection_name: 
        :param points: 
        :return: 
        """
        return self.qdrant_client.upsert(collection_name, points)
    
    def retrieve(self, collection_name, ids):
        """
        从集合中检索点。
        :param collection_name: 
        :param ids: 
        :return: 
        """
        return self.qdrant_client.retrieve(collection_name, ids)
    
    def delete(self, collection_name, points_selector):
        """
        从集合中删除点。
        :param collection_name: 
        :param points_selector: 
        :return: 
        """
        return self.qdrant_client.delete(collection_name, points_selector)
    
    def set_payload(self, collection_name, payload, points):
        """
        为点设置有效负载。
        :param collection_name: 
        :param payload: 
        :param points: 
        :return: 
        """
        return self.qdrant_client.set_payload(collection_name, payload, points)
    
    def overwrite_payload(self, collection_name, payload, points):
        """
        为点设置有效负载。
        :param collection_name: 
        :param payload: 
        :param points: 
        :return: 
        """
        return self.qdrant_client.overwrite_payload(collection_name, payload, points)
    
    

if __name__ == '__main__':
    qdrant_client_demo = QdrantClientDemo()
    qdrant_client = qdrant_client_demo.get_qdrant_client()
    print(qdrant_client_demo.get_collections())



