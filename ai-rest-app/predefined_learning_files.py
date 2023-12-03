from enum import Enum


LEARNING_FILES_ROOT_DIR = "./datasets"


class PredefinedLearningFile(str, Enum):

    def path(self):
        return f'{LEARNING_FILES_ROOT_DIR}/{self.value}'

    BITCOIN = "BitcoinUSD.csv",
    ETHERNEUM = "EtherUSD.csv",
    ACALA = "AcalaUSD.csv",
    HARVEST = "HarvestUSD.csv"
