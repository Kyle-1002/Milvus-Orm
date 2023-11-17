from milvus.constant import DbConstant


def table(name: str, description: str = ""):
    """

    :param name: 表名
    :param description: 表描述
    :return:
    """

    def decorator(cls):
        setattr(cls, DbConstant.TABLE_NAME, name)
        setattr(cls, DbConstant.TABLE_DESCRIPTION, description)
        return cls

    return decorator
