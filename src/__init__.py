import pyexpat  # noqa: F401  (must load before RDKit's drawing module, whose bundled expat crashes later XML parsing)

from src.utils import hydra_resolvers  # noqa: F401  (registers the `hydra_or` resolver)
