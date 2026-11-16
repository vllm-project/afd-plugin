import torch

# FRAGMENT coexists with the CAM extension and never registers CANN common ops.
# ### PATCH START: share AFD namespace without redefining common operators
library = torch.library.Library("afd_ascend", "FRAGMENT")
# ### PATCH END: share AFD namespace without redefining common operators
