from enum import Enum


LEARNING_FILES_ROOT_DIR = "./datasets"


class PredefinedLearningFile(str, Enum):

    def path(self):
        return f'{LEARNING_FILES_ROOT_DIR}/{self.value}'

    AIRLINE_PASSENGERS = "airline-passengers.csv",
    KOPIA = "kopia.csv",
    ZBIOR_DRUGI = "zbior-drugi.csv",
    ZBIOR_TRZECI = "zbior-trzeci.csv"
