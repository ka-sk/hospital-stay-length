import torch
import models

def filepath(model: torch.nn.Module, cv_num: int):
    print(model._get_name)
    pass


def file_name():
    pass


if __name__ == '__main__':
    model_list = models.get_all_models()
    filepath(model_list[50], 1)
