from Metric import Viallina, ModelWithTemperature, ModelActivateDropout,Mahalanobis
from BasicalClass import Fashion_Module
import torch

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
if device.type != 'cpu':
    torch.cuda.set_device(device=device)
module = Fashion_Module(device, load_poor = False)

# v = Viallina(module, device)
#
# t = ModelWithTemperature(module,device)
# t.set_temperature(module.val_loader)
# t.run_experiment(module.val_loader, module.test_loader)
#
#
# mc = ModelActivateDropout(module, device, iter_time= 500)
# mc.run_experiment(module.val_loader, module.test_loader)


ma = Mahalanobis(module, device)
ma.run_experiment(module.val_loader, module.test_loader)