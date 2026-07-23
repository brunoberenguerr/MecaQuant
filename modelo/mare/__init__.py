"""MARÉ — Mean-reversion Adaptive Residual Engine.

Estratégia de reversão à média sobre retornos residuais de ações do S&P 500.

Tese econômica: reversão de curto prazo é prêmio por prover liquidez a fluxo
impaciente, e é maior quando o capital de intermediação está restrito
(Nagel 2012; Grossman & Miller 1988).
"""

from mare.config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
