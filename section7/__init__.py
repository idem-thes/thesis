"""code_section7 - Section 7 (Forecasting) implementation package.

"""

import logging
import os

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
