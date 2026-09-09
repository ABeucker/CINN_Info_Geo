import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import sys
import logging
logger = logging.getLogger(__name__)

# Invertible Flow model for inference, adapted from https://github.com/astro-ML/21cm_pie

# Activation functions available for the coupling subnetworks, selected via params['act'].
ACTIVATIONS = {
    'relu': lambda: nn.ReLU(),
    'leakyrelu': lambda: nn.LeakyReLU(0.2),
    'elu': lambda: nn.ELU(),
    'selu': lambda: nn.SELU(),
    'gelu': lambda: nn.GELU(),
}

class ConditionalInvertibleBlock():
    """
    Conditional Invertible Block originally for EoRFlow.

    Args:
        params (dict): A dictionary containing the parameters for the block.
            Relevant here: params['act'] selects the subnet activation --
            'relu' (default), 'leakyrelu', 'elu', 'selu', or 'gelu'.
            params['coupling'] selects the coupling block type -- 'affine'
            (default, Fm.AllInOneBlock) or 'spline' (Fm.RationalQuadraticSpline
            + an explicit permutation between blocks).

    Attributes:
        params (dict): A dictionary containing the parameters for the block.
        flow (Ff.SequenceINN): The flow model.

    Methods:
        __init__(self, params: dict) -> None: Initializes the ConditionalInvertibleBlock object.
        model(self, n_dim: int, n_blocks: int, n_nodes: int, cond_dims: tuple) -> Ff.SequenceINN: Constructs the flow model.
        load_model(self, location: str): Loads the model from the specified location.
    """

    def __init__(self, params: dict) -> None:
        """
        Initializes the ConditionalInvertibleBlock object.

        Args:
            params (dict): A dictionary containing the parameters for the block.
        """
        self.params = params
        n_dim = params['n_dim']
        n_blocks = params['n_blocks']
        n_nodes = params['n_nodes']
        subnet_depth = params['subnet_depth']
        self.cond_dims = params['cond_dims']
        self.flow = self.model(n_dim, n_blocks, n_nodes, subnet_depth, self.cond_dims)
        if self.params['load']:
            if self.load_model(self.params['model_location']):
                logger.info(f"Loaded flow from {self.params['model_location']}")
            else:
                logger.info(f"Failed to load flow from {self.params['model_location']}")
                sys.exit()


    def model(self, n_dim: int, n_blocks: int, n_nodes: int, subnet_depth: int, cond_dims: int) -> Ff.SequenceINN:
        """
        Constructs the flow model.

        Args:
            n_dim (int): The dimensionality of the input.
            n_blocks (int): The number of blocks in the model.
            n_nodes (int): The number of nodes in the subnet.
            cond_dims (tuple): The dimensions of the conditional input.

        Returns:
            Ff.SequenceINN: The constructed flow model.
        """

        def subnet_fc(dims_in: int, dims_out: int) -> nn.Sequential:
            act_name = self.params.get('act', 'relu').lower()
            if act_name not in ACTIVATIONS:
                raise ValueError(f"Unsupported activation function: {self.params.get('act')}")

            layers = []
            input_dim = dims_in
            for _ in range(subnet_depth):
                layers.append(nn.Linear(input_dim, n_nodes))
                layers.append(ACTIVATIONS[act_name]())
                input_dim = n_nodes
            layers.append(nn.Linear(input_dim, dims_out))
            return nn.Sequential(*layers)
        
        flow = Ff.SequenceINN(n_dim)
        permute_soft = True if self.params['n_dim'] != 1 else False
        coupling = self.params.get('coupling', 'affine').lower()

        for k in range(n_blocks):
            if coupling == 'affine':
                flow.append(Fm.AllInOneBlock, cond=0, cond_shape=(cond_dims,),
                            subnet_constructor=subnet_fc, permute_soft=permute_soft)
            elif coupling == 'spline':
                flow.append(Fm.RationalQuadraticSpline, cond=0, cond_shape=(cond_dims,),
                            subnet_constructor=subnet_fc)
                if n_dim != 1:
                    flow.append(Fm.PermuteRandom)
            else:
                raise ValueError(f"Unsupported coupling type: {self.params.get('coupling')}")
        return flow
    
    def load_model(self, location: str) -> bool:
        """
        Loads a pre-trained model from the specified location.

        Parameters:
        - location (str): The file path of the pre-trained model.

        Returns:
        - bool: True if the model is successfully loaded, False otherwise.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.flow.load_state_dict(torch.load(location, map_location=device))
            return True
        except FileNotFoundError:
            return False
    
    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of x given conditioning variables c.

        Args:
            x (torch.Tensor): Samples for which to compute log probabilities. Shape: [batch_size, n_dim]
            c (torch.Tensor): Conditioning variables. Shape: [batch_size, cond_dims]

        Returns:
            torch.Tensor: Log probabilities of the samples. Shape: [batch_size]
        """

        z, jac = self.flow(x, c=[c], rev=False)
        
        log_prob = -0.5 * torch.sum(z **2, dim=1) + jac
        
        return log_prob

    # --- Adapter-like helpers ---
    def parameters(self):
        """Return parameters iterator for optimizer compatibility."""
        return self.flow.parameters()

    def to(self, device):
        """Move underlying FrEIA flow to device."""
        self.flow.to(device)
        return self

    def state_dict(self):
        return self.flow.state_dict()

    def load_state_dict(self, sd):
        return self.flow.load_state_dict(sd)
    
    def half(self):
        self.flow.half()
        return self

    def batch_loss(self, theta: torch.Tensor, cond: torch.Tensor):
        """
        Compute the flow loss .

        Expects `theta` to already be on the unconstrained real line (e.g. logit of xHI)
        and `cond` to be the conditioning vector.
        Returns a scalar loss (mean over batch).
        """
        # ensure cond shape: (batch, cond_dims)
        if cond.shape != torch.Size([theta.size(0), self.cond_dims]):
            cond = cond.repeat(theta.size(0), 1)
        
        z, jac = self.flow(theta, c=[cond], rev=False)
        loss = (0.5 * z.pow(2).sum(1) - jac).mean() / float(self.params['n_dim'])
        return loss

    def sample(self, n_samples: int, cond: torch.Tensor):
        """
        Sample from the conditional flow: z ~ N(0,I) then x = f^{-1}(z|cond).

        Returns samples z ~ N(0,I) and x on the unconstrained real line (you may need to sigmoid them
        externally if targets live in (0,1)).
        """
        device = next(self.flow.parameters()).device
        dtype = next(self.flow.parameters()).dtype
        z = torch.randn(n_samples, self.params['n_dim'], device=device, dtype=dtype)
        # STILL HAVE TO FIX THIS! See commented out code below, need to include possibility that cond is already batched 
        cond_rep = cond.repeat(n_samples, 1)

        x, _ = self.flow(z, c=[cond_rep], rev=True)
        return z, x
    
        # if cond.ndim == 1:
        #     cond_rep = cond.unsqueeze(0).repeat(n_samples, 1)
        # else:
        #     cond_rep = cond