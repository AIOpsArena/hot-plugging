import json
import logging
import os
import subprocess
import sys
import pathlib
darin3_path = pathlib.Path(__file__).parent
sys.path.append(str(darin3_path))
DF_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(DF_path))
import time
from os.path import dirname
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from DiagFusion import processed_data_config
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


# in_log_file = "../data/log/log_logstash-service_1.csv"
def log_drain(in_log_file):
    config = TemplateMinerConfig()
    config.load("drain3.ini")
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    line_count = 0

    with open(in_log_file) as f:
        lines = f.readlines()

    start_time = time.time()
    batch_start_time = start_time
    batch_size = 10000
    result_json_list = []
    for line in lines:
        line = line.rstrip()
        line = line.split(',', 5)[-1]
        result = template_miner.add_log_message(line)
        line_count += 1
        if line_count % batch_size == 0:
            time_took = time.time() - batch_start_time
            rate = batch_size / time_took
            logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                        f"{len(template_miner.drain.clusters)} clusters so far.")
            batch_start_time = time.time()
        if result["change_type"] != "none":
            result_json = json.dumps(result)
            logger.info(f"Input ({line_count}): " + line)
            logger.info("Result: " + result_json)
        result_json_list.append(result)

    time_took = time.time() - start_time
    rate = line_count / time_took
    logger.info(
        f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
        f"{len(template_miner.drain.clusters)} clusters")

    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
    for cluster in sorted_clusters:
        logger.info(cluster)

    with open(os.path.join(processed_data_config['drain3_result_dir'], 'logstash_structured.json'), 'w+') as f:
        json.dump(result_json_list, f)

    with open(os.path.join(processed_data_config['drain3_result_dir'], 'logstash_templates.json'), 'w+') as f:
        for cluster in sorted_clusters:
            f.write(str(cluster))
            f.write('\n')

    print("Prefix Tree:")
    template_miner.drain.print_tree()

    template_miner.profiler.report(0)