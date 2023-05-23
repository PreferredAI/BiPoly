import configparser
import logging.config

config = configparser.ConfigParser()
config.read(".config.ini")
logging.config.fileConfig(".config.ini")

OUTPUT_DIR = config["app"]["output_dir"]
if not OUTPUT_DIR.endswith("/") or not OUTPUT_DIR.endswith("\\"):
    OUTPUT_DIR = OUTPUT_DIR + "/"
CONNECTIONS = int(config["app"]["connections"])

STEM_BATCH = int(config["stem"]["batch"])

DISTRIBUTED_LOCAL = "true" == config["distributed"]["local"].lower()
DISTRIBUTED_IP = config["distributed"]["ip"]

DISTRIBUTED_CORES = int(config["distributed"]["cores"])
DISTRIBUTED_THREADS = int(config["distributed"]["threads"])
DISTRIBUTED_MEMORY = config["distributed"]["memory"]
DISTRIBUTED_WORKERS = int(config["distributed"]["workers"])
DISTRIBUTED_WORKERS_MIN = int(config["distributed"]["workers_min"])

DISTRIBUTED_JOB_CPU = float(config["distributed"]["job_cpu"])
DISTRIBUTED_JOB_MEM = config["distributed"]["job_mem"]
DISTRIBUTED_JOB_QUEUE = config["distributed"]["job_queue"]

DISTRIBUTED_DIR = config["distributed"]["dir"]
