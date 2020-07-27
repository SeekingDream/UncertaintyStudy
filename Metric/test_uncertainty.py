from Metric import *
from utils import Fashion_Module
import torch


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
module = Fashion_Module(device=device)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# v = Viallina(module, device)
# v.run()
# v = ModelWithTemperature(module, device)
# v.run()

v = Mutation(module, device, it_time=2)
v.run()


