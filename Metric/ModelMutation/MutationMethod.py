from utils import *
from Metric import BasicUncertainty
from Metric.ModelMutation import GaussianFuzzing, NeuronActivationInverse, WeightShuffling, NeuronSwitch


class Mutation(BasicUncertainty):
    name_list = [
        'GaussianFuzzing',
        'WeightShuffling',
        'NeuronSwitch',
        'NeuronActivationInverse',
    ]

    def __init__(self, instance: BasicModule,  device, it_time):
        super(Mutation, self).__init__(instance, device)
        self.it_time = 2 if IS_DEBUG else it_time
        self.gf = GaussianFuzzing(self.instance.model, device=self.device)
        self.nai = NeuronActivationInverse(self.instance.model, device=self.device)
        self.ws = WeightShuffling(self.instance.model, device=self.device)
        self.ns = NeuronSwitch(self.instance.model, device=self.device)
        self.op_list = [self.gf, self.nai, self.ws, self.ns]

    @staticmethod
    def label_chgrate(orig_pred, prediction):
        return np.sum(orig_pred.reshape([-1, 1]) == prediction, axis=1)

    def _uncertainty_calculate(self, data_loader):
        score_list = []
        _, orig_pred, _ = common_predict(data_loader, self.model, self.device)
        orig_pred = common_ten2numpy(orig_pred)
        for op in self.op_list:
            print(op.__class__.__name__)
            mutation_matrix = op.run(data_loader, iter_time=self.it_time)
            score = self.label_chgrate(orig_pred, mutation_matrix)
            score_list.append(score)
        return score_list
