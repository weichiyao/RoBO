import torch
from torch import Tensor
from functorch import vmap

class Transformer(object):
    """
    Transform with radially symmetric fields
    - Rescale all the input domain to be [0, 1]
    - Centers are equally spaced with delta 
    
    For any given input, apply
    - standardization
    - radial basis function transformation
    
    Arguments
    ----------
    search_domain : numpy array (d,2)
    n_rad : int
        number of centers on each dimension
    prodsum: bool=False
        Whether to use product sum formulation; 
        otherwise, just the union sum formulation
        If used, n_rad**d centers in total; if not, n_rad*d.
    x : torch tensor (..., d) 
        n number of d-dimensional features to be transformed 
    
    Returns
    ----------
    x : torch tensor (..., nrad**d)
        transformed features
    """
    def __init__(
        self, 
        search_domain: Tensor,
        method: str="none",
        n_rad: int=50,
        prodsum: bool=False,
        dtype: torch.float32
    ):  
        self.search_domain = search_domain 
        self.n_rad = n_rad # nrad for each dimension 
        self.prodsum = prodsum 
        self.d = search_domain.shape[0]
        self.dtype = dtype
        
        self.n_features = self.d 
        self.method = method
        if self.method == "rbf":
            # compute the parameters for radial basis function transform
            self._get_rbf_params() 
        elif self.method == "std":
            # compute the parameters for standardization
            self._get_std_params()
        elif self.method == "none":
            pass 
        else:
            raise ValueError(f"Transformer method {self.method} is not implemented.")  
            
        self._TRANSFORMER = {
            "none": lambda x: x,
            "std" : self._standardize,
            "rbf" : self._rbf_transform  
        }
        
    def _get_std_params(self): 
        """
        For x in range [a, b], we standardize x by 
        (x - a) / (b - a)
        """
        self.standardize_m = self.search_domain[:,0]
        self.standardize_s = self.search_domain[:,1]-self.search_domain[:,0]  
    
    def _standardize(self, x:Tensor) -> Tensor:
        """Max-Min standardization

        Arguments:
        ===================
        x: torch tensor (...,d)
           Features to be standardized in each dimension
        m: torch tensor (d,)
           Min value in each dimension 
        s: torch tensor (d,)
           Range in each dimension

        Returns:
        ===================
        x: torch tensor (...,d)
           Standardized features
        """
        return (x - self.standardize_m.to(x.device)) / self.standardize_s.to(x.device)
    
    def _get_rbf_params(self):
        """
        here we assume all have been standardized 
        and x values are all within [0,1]
        
        Arguments:
        ===================
        prodsum: bool=False
            Whether to use product sum formulation; 
            otherwise, just the union sum formulation
            If used, n_rad**d centers in total; if not, n_rad*d.

        Returns:
        ===================
        mu: (n_rad, d) 
        gamma: float
        """ 
        gamma = torch.stack([2*(xmax - xmin)/(self.n_rad - 1) 
                             for (xmin, xmax) in self.search_domain], dim=0)
        mu = torch.stack([torch.linspace(start=xmin, end=xmax, steps=self.n_rad)
                          for (xmin, xmax) in self.search_domain], dim=-1)
        
        self.rbf_mu = mu.to(self.dtype)
        self.rbf_gamma = gamma.to(self.dtype)

        if self.prodsum:
            self.n_features = self.n_rad ** self.d
        else:
            self.n_features = self.n_rad * self.d

    def _map_rbf_func(self, x:Tensor, mu:Tensor, gamma:Tensor) -> Tensor:
        """ 
        param x: (...,d)
            Feature value
        param mu: (n_rad,d)
            Radial basis centers
        param gamma (d,)
        returns gamma * ||x - mu||^2: (n, d, n_rad) 
        """
        return gamma * (x.unsqueeze(-1) - mu.unsqueeze(0)) ** 2 # (n, d, n_rad) 
    
    def _rbf_transform(self, x:Tensor) -> Tensor:
        """Radial basis function transform

        Arguments:
        ===================
        x: (...,d)
            Feature value to be transformed
        mu: (n_rad,d)
            Radial basis centers
        gamma: (d,)

        Returns:
        ===================
        x: (..., n_rad**d) if prodsum else (..., n_rad*d) 
           Transformed features 
        """ 
        
        self.rbf_mu    = self.rbf_mu.to(x.device)
        self.rbf_gamma = self.rbf_gamma.to(x.device)

        out = vmap(self._map_rbf_func, in_dims=-1, out_dims=1)(x, self.rbf_mu, self.rbf_gamma) # (n, d, n_rad)

        if self.prodsum:
            out = vmap(lambda y: torch.cartesian_prod(*y).sum(-1), in_dims=0)(out) # (n, n_rad**d)
        else:
            out = out.reshape(-1, self.n_features)  # (n, n_rad*d)

        if x.dim() == 1:
            out = out.squeeze(0)
        return torch.exp(-out)
                
        
    def __call__(self, x:Tensor) -> Tensor:
        """
        Input x : torch tensor (..., d)
            -- n number of d-dimensional features to be transformed 
        Output x : torch tensor (..., n_features)
            -- n_features depends on which transform method is used
        """ 
        return self._TRANSFORMER[self.method](x) 

 
