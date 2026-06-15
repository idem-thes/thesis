"""Intraday VIX-fit experiment - eq 53 Euler vs standard 2EXP regression.

Tests whether feeding hourly SPX returns through the Ito-derived dsigma_t SDE
(thesis eq 53, lines 712-741) tracks daily VIX_close better than the
plain regression sigma_t = beta_0 + beta_1 R_1 + beta_2 sqrt.R_2 evaluated once per day.
"""

import logging

from . import _paths 

log = logging.getLogger(__name__)
