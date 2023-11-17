class BaseModel:
    @classmethod
    def get_instance(cls, **kwargs):
        instance = cls.__new__(cls)
        for k, v in kwargs.items():
            setattr(instance, k, v)
        return instance


if __name__ == "__main__":
    model = BaseModel.get_instance(id=1, value=2)
    print(model)
