import torch


class Tool:
    @staticmethod
    def get_gpu():
        """如果有GPU则返回gpu(cuda)设备，否则返回cpu"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def soft_update(_from, _to):
        for _from, _to in zip(_from.parameters(), _to.parameters()):
            value = _to.data * 0.99 + _from.data * 0.01
            _to.data.copy_(value)

    @staticmethod
    def requires_grad(model, value):
        for param in model.parameters():
            param.requires_grad_(value)
