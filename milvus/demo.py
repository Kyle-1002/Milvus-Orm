from milvus.field import Field
from milvus.annotation import table
from milvus.model import BaseModel

from milvus.repository import Repository

table_name = "test_chat_message"


# 使用table注解注明表信息
# 所有表的model继承BaseModel
# Model类是数据表的映射 Model实例是数据行的映射
@table(name=table_name, description="have a test")
class ChatMessageModel(BaseModel):
    # id是自增主键 实例的时候不需要设置 设置了也没有用
    # name是数据表里字段的名字
    id = Field.LongField(name="id", is_primary=True, auto_id=True)
    msg_type = Field.IntField("msg_type")
    sender = Field.StringField("sender")
    receiver = Field.StringField("receiver")
    create_time = Field.LongField("create_time")
    update_time = Field.LongField("update_time")
    content = Field.StringField("content")
    # dim是向量的维度
    vector = Field.FloatField(name="vector", dim=8)


def demo():
    # 初始化milvus的仓库 连接数据库
    repo = Repository(alias="default", host="http://127.0.0.1/", port="19530")
    # 创建数据表
    table = repo.create_table(ChatMessageModel)
    # 如果已经创建 可以直接获取数据表
    table = repo.get_table(table_name)
    # 实例化插入的数据
    model = ChatMessageModel.get_instance(id=1, msg_type=1, sender="derrick", receiver="kyle")
    # 插入数据
    repo.insert(model)
    models = [model, model]
    # 批量插入数据
    repo.batch_insert(models)
    # 普通检索数据
    query_model = ChatMessageModel.get_instance(msg_type=1)
    res = repo.search(query_model)
    # 向量检索数据
    # 第二个参数传入需要进行向量检索的字段
    search_model = ChatMessageModel.get_instance(vector=[0 for i in range(8)])
    res = repo.search(search_model, ChatMessageModel.vector)
    # 向量检索+普通检索
    search_query_model = ChatMessageModel.get_instance(msg_type=1, vector=[0 for i in range(8)])
    res = repo.search(search_query_model, ChatMessageModel.vector)
    print(res)


if __name__ == "__main__":
    demo()
