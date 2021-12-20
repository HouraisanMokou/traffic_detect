import torch.nn as nn
import torch.functional as F

class RPN_PROPOSAL_TARGET(nn.Module):
    """
    proposal layer of rpn net
    """