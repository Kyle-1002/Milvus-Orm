from pymilvus import DataType


class Field(object):
    class BaseField(object):
        pass
    class VectorField(BaseField):
        pass

    class StringField(BaseField):
        def __init__(self, name: str, max_length:int = 128, description: str = ""):
            self.name = name
            self.dtype = DataType.VARCHAR
            self.default = ""
            self.description = description
            self.max_length = max_length

    class IntField(BaseField):
        def __init__(self, name: str, description: str = ""):
            self.name = name
            self.dtype = DataType.INT64
            self.default = 0
            self.description = description


    class LongField(BaseField):
        """
        auto_id为true的时候，实例不用设置这个字段 设置了也没用
        """
        def __init__(self, name: str, is_primary: bool = False, auto_id: bool = False, description: str = ""):
            self.name = name
            self.dtype = DataType.INT64
            self.default = 0
            self.is_primary = is_primary
            self.auto_id = auto_id
            self.description = description

    class BoolField(BaseField):
        def __init__(self, name: str, description: str = ""):
            self.name = name
            self.dtype = DataType.BOOL
            self.default = True
            self.description = description

    class BinaryField(VectorField):
        def __init__(self, name: str, dim: int, description: str = ""):
            self.name = name
            self.dtype = DataType.BINARY_VECTOR
            self.dim = dim
            self.default = []
            self.description = description

    class FloatField(VectorField):
        def __init__(self, name: str, dim: int, description: str = ""):
            self.name = name
            self.dtype = DataType.FLOAT_VECTOR
            self.dim = dim
            self.default = [0.0 for i in range(dim)]
            self.description = description
