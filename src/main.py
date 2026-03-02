from src.config.config import Config
import src.utils.logger as logger
import sys
from src.preprocessing.preprocessor import Preprocessor

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <data_path>")
        sys.exit(1)

    config = Config()
    logger.setup_logging(config.environment)

    data_path = sys.argv[1]
    preprocessor = Preprocessor(data_path)
