import os

PATH_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PATH_DATASETS = os.environ["PYG_DATASETS"] or PATH_PROJECT_ROOT + "/data"
