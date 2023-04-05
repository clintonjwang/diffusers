import pdb
from functools import partial
import torch
F = torch.nn.functional
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from tqdm import trange
from typing import Any, Dict, Callable, List, Optional, Union

class SDSStableDiffusionPipeline(StableDiffusionPipeline):
    pass