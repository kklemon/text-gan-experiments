import copy

from text_gans.utils import update_average


class SamplingModel:
    def __init__(self, model):
        self.model = model

    def update(self, model):
        self.model = model


class EMASamplingModel(SamplingModel):
    def __init__(self, model, beta=0.999):
        super().__init__(copy.deepcopy(model))
        update_average(self.model, model, beta=0.0)

        self.beta = beta

    def update(self, model):
        update_average(self.model, model, self.beta)


def get_sampling_model(model, use_ema: bool):
    if use_ema:
        return EMASamplingModel(model)
    return SamplingModel(model)
