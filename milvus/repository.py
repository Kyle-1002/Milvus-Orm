from pymilvus import connections, CollectionSchema, utility
from pymilvus import Collection, FieldSchema

from milvus.constant import DbConstant
from milvus.field import Field
from milvus.model import BaseModel


class Repository:

    def __init__(self, alias="default", host="http://127.0.0.1/", port="19530"):
        """

        :param alias: 用于关闭连接的别名
        :param host:
        :param port:
        """
        self.alias = alias
        self.host = host
        self.port = port
        connections.connect(alias, host, port)

    def create_table(self, cls):
        """
        创建数据表
        :param cls: 数据表的Model类
        :return: Collection
        """
        if not issubclass(cls, BaseModel):
            raise TypeError("Model必须是BaseModel的子类")
        table_name = getattr(cls, DbConstant.TABLE_NAME)
        if utility.has_collection(table_name):
            return self.get_table(table_name)
        sorted_fields = []
        for k, v in cls.__dict__.items():
            if isinstance(v, Field.BaseField):
                sorted_fields.append(k)
        sorted_fields.sort()
        schemas = []
        for field in sorted_fields:
            schema_args = cls.__dict__.get(field)
            schema = FieldSchema(**schema_args.__dict__)
            schemas.append(schema)

        table_description = getattr(cls, DbConstant.TABLE_DESCRIPTION)
        table_schema = CollectionSchema(
            fields=schemas,
            description=table_description
        )
        collection = Collection(name=table_name,
                                schema=table_schema,
                                using='default', )
        return collection

    def get_table(self, name: str):
        """
        根据名字获取数据表
        :param name: 表名
        :return: Collection
        """
        if utility.has_collection(name):
            collection = Collection(name)
            return collection
        return None

    def delete_table(self, name: str):
        """
        根据名字删除数据表
        :param name: 表名
        :return: void
        """
        if utility.has_collection(name):
            utility.drop_collection(name)

    def insert(self, model: BaseModel):
        """
        插入数据行
        :param model: 数据表的Model实例
        :return:
        """
        cls = type(model)
        if not issubclass(cls, BaseModel):
            raise TypeError("Model必须是BaseModel的子类")
        table_name = getattr(model, DbConstant.TABLE_NAME)
        collection = self.get_table(table_name)
        if not collection:
            raise RuntimeError("数据表 {} 不存在".format(table_name))
        sorted_fields = []
        for k, v in cls.__dict__.items():
            if isinstance(v, Field.BaseField):
                # 如果是自增主键 不要赋值
                if isinstance(v, Field.LongField) and getattr(v, DbConstant.AUTO_ID):
                    continue
                sorted_fields.append(k)
                # 实例没有赋值的时候 会拿到类的属性 也就是BaseField 如果不是auto_id 需要给个默认值
                instance_value = getattr(model, k)
                if isinstance(instance_value, Field.BaseField):
                    setattr(model, k, getattr(v, DbConstant.DEFAULT_KEY))
        sorted_fields.sort()
        data = [[getattr(model, field)] for field in sorted_fields]
        collection.insert(data)

    def batch_insert(self, models: list):
        """
        批量插入数据行 当前只支持一个数据表的批量
        :param models: 数据表的Model实例列表
        :return:
        """
        if not isinstance(models, list):
            raise TypeError("Models必须是List")
        if len(models) == 0:
            return
        cls = type(models[0])
        table_name = getattr(models[0], DbConstant.TABLE_NAME)
        collection = self.get_table(table_name)
        if not collection:
            raise RuntimeError("数据表 {} 不存在".format(table_name))
        for model in models:
            model_cls = type(model)
            if not issubclass(model_cls, BaseModel):
                raise TypeError("Model必须是BaseModel的子类")
            if cls != model_cls:
                raise TypeError("List里的Model必须统一类型")
        sorted_fields = []
        for k, v in cls.__dict__.items():
            if isinstance(v, Field.BaseField):
                # 如果是自增主键 不要赋值
                if isinstance(v, Field.LongField) and getattr(v, DbConstant.AUTO_ID):
                    continue
                sorted_fields.append(k)
                for model in models:
                    # 实例没有赋值的时候 会拿到类的属性 也就是BaseField 如果不是auto_id 需要给个默认值
                    instance_value = getattr(model, k)
                    if isinstance(instance_value, Field.BaseField):
                        setattr(model, k, getattr(v, DbConstant.DEFAULT_KEY))
        sorted_fields.sort()
        data = [[getattr(model, field) for model in models] for field in sorted_fields]
        collection.insert(data)

    def search(self, model: BaseModel, search_field: Field.VectorField = None, metric_type: str = "L2",
               limit: int = 10):
        """

        :param model: 普通参数是equal查询 向量参数是向量检索
        :param search_field: 需要向量检索的字段
        :param metric_type: 距离参数
        :param limit: 查询个数
        :return:
        """
        model_cls = type(model)
        if not issubclass(model_cls, BaseModel):
            raise TypeError("Model必须是BaseModel的子类")
        if search_field:
            search_field_cls = type(search_field)
            if not issubclass(search_field_cls, Field.VectorField):
                raise TypeError("Field必须是Field.VectorField的子类")

        search_params = {"metric_type": metric_type}

        table_name = getattr(model, DbConstant.TABLE_NAME)
        data = None
        anns_field = None
        expressions = []
        output_fields = []
        primary_key = ""
        for k, v in model_cls.__dict__.items():
            if not isinstance(v, Field.BaseField):
                continue
            if isinstance(v, Field.VectorField):
                if search_field and v.name == search_field.name:
                    model_search_value = getattr(model, k)
                    if not isinstance(model_search_value, Field.BaseField):
                        data = [model_search_value]
                    anns_field = v.name
                continue
            if isinstance(v, Field.LongField) and v.is_primary:
                primary_key = v.name
            output_fields.append(v.name)
            filter_value = getattr(model, k)
            if isinstance(filter_value, Field.BaseField):
                continue
            expression = v.name + "==" + str(filter_value)
            expressions.append(expression)

        expr = None if len(expressions) == 0 else " and ".join(expressions)
        table = self.get_table(table_name)
        # todo 优化性能
        table.load()
        if data and anns_field:
            results = table.search(data, anns_field,
                                   param=search_params,
                                   output_fields=output_fields,
                                   limit=limit,
                                   expr=expr,
                                   consistency_level="Strong")
            result = results[0]
            if len(result.ids) == 0:
                return None
            distance_map = {result.ids[i]: result.distances[i] for i in range(len(result.ids))}
            result_ids_expression = ",".join([str(id) for id in result.ids])
            query_expression = primary_key + " in [ " + result_ids_expression + " ]"
            query_results = table.query(expr=query_expression,
                                        output_fields=output_fields,
                                        consistency_level="Strong")
            models = []
            for query_result in query_results:
                model = model_cls.get_instance(**query_result)
                setattr(model, DbConstant.DISTANCE, distance_map[getattr(model, primary_key)])
                models.append(model)
            table.release()
            return models

        elif expr:
            results = table.query(
                expr=expr,
                output_fields=output_fields,
                consistency_level="Strong")
            models = []
            for result in results:
                model = model_cls.get_instance(**result)
                models.append(model)
            table.release()
            return models
        else:
            raise RuntimeError("没有筛选条件")

        def __del__(self):
            connections.disconnect(self.alias)
