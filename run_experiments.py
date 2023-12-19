from load_config import *
from Mawps import *


# show configuration values
cfg_g2t = get_args()
print(cfg_g2t)
# import model
model = Mawps(cfg_g2t)
# train model and record accuracy
best_acc = model.train()
