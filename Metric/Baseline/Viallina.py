from BasicalClass import BasicModule
from BasicalClass import common_get_maxpos, common_predict
from Metric import BasicUncertainty


class Viallina(BasicUncertainty):
    def __init__(self, instance: BasicModule, device):
        super(Viallina, self).__init__(instance, device)

    def _uncertainty_calculate(self, data_loader):
        pred_pos, _, _ = common_predict(data_loader, self.model, self.device)
        return common_get_maxpos(pred_pos)


