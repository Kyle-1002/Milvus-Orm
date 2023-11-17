from pymilvus.orm import utility

from milvus.demo import ChatMessageModel
from milvus.repository import Repository

if __name__ == "__main__":
    table_name = "test_chat_message"
    repo = Repository()
    # 测试通过
    # table = repo.create_table(ChatMessageModel)
    collections = utility.list_collections()
    # 测试通过
    # repo.delete_table(table_name)
    collection = repo.get_table(table_name)
    model = ChatMessageModel.get_instance(msg_type=1, sender="kyle", receiver="derrick", create_time=0)
    # 测试通过
    # repo.insert(model)
    model2 = ChatMessageModel.get_instance(msg_type=2, sender="royce")
    model_list = [model, model2]
    # 测试通过
    # repo.batch_insert(model_list)
    collection = repo.get_table(table_name)
    if collection:
        search_model = ChatMessageModel.get_instance(vector=[0 for i in range(8)])
        # 测试通过
        # search_result = repo.search(search_model, ChatMessageModel.vector)

        query_model = ChatMessageModel.get_instance(msg_type=1)
        # 测试通过
        # query_result = repo.search(query_model)

        search_query_model = ChatMessageModel.get_instance(msg_type=1, vector=[0 for i in range(8)])
        # 测试通过
        # search_query_result = repo.search(search_query_model, ChatMessageModel.vector)
        print("")

