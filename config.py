from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(message)s", 
                    datefmt="%m/%d/%Y %H:%M:%S",
                    # filename='xxx.log',
                    # filemode='a'
                    )

@dataclass
class Config:
    logger = logging.getLogger()


config = Config()