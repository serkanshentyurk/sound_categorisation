  """
  models — BE and SC computational models for inference.

  Usage:
      from models import BEParams, BEState, BEModel
      from models import SCParams, SCState, SCModel
  """
  from models.BE_core import BEParams, BEState, BEModel, ModelTrace
  from models.SC_core import SCParams, SCState, SCModel
  from models.perception import perceive_stimulus 

  __all__ = [
      'BEParams', 'BEState', 'BEModel',
      'SCParams', 'SCState', 'SCModel',
      'ModelTrace',
      'perceive_stimulus',
  ]