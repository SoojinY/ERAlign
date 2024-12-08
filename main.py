from utils import (
    Dataset,
    Refiner,
    Verifier,
    eval_sims,
    sinkhorn,
    new_rank_results,
    verified_dist,
)
import argparse
import os
import logging 
import sys
# Set log
logging.basicConfig(format="%(asctime)s [%(levelname)s]%(name)s: %(message)s",datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.DEBUG,
            handlers=[logging.StreamHandler(sys.stdout)],)
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

#choose the GPU, "-1" represents using the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--datapath", type=int, default=0, 
    help="Path of Dataset. 0: DBP15k/zh_en/, 1: DBP15k/ja_en/, 2: DBP15k/fr_en/, 3: SRPRS/FR_EN/, 4: SRPRS/DE_EN/"
    )
parser.add_argument(
    "--mode", type=str, default="hybrid", 
    help="\"hybrid\": Semantic + Lexical or \"sem\": Semantic"
    )
parser.add_argument(
    "--iters", type=int, default=1, 
    help="Number of iteration for fusing similarity scores of entities and relations"
    )
parser.add_argument("--tau", type=float, default=0.2)
parser.add_argument("--lam", type=float, default=0.5, help="Lambda for fusing")
args = parser.parse_args()

logger.info(args)

mode = args.mode.lower()
data_path = "data/" + ["DBP15K/zh_en/", "DBP15k/ja_en/", "DBP15k/fr_en/", "SRPRS/FR_EN/", "SRPRS/DE_EN/"][args.datapath]

dataset = Dataset(data_path, mode)
refiner = Refiner(args.iters, args.tau, args.lam)

for i in range(args.iters + 1):
    logger.info(f"Refinement iters : {i}/{args.iters}")
    # if i==0:
    refiner.ent_sims = refiner.refine_sims(dataset, target="ent", iters=i)
    if i > 0:
        eval_sims(sinkhorn(refiner.ent_sims), dataset)
    else:
        eval_sims(refiner.ent_sims, dataset)
    refiner.rel_sims = refiner.refine_sims(dataset, target="rel", iters=i)

verifier = Verifier(dataset, refiner)
verify_ranks, verified_idx = verifier.verify(dataset)
new_ranks, new_results = new_rank_results(verifier, dataset, verify_ranks) 
verified_case = verified_dist(verifier, new_results, verified_idx)

logger.info("Done")
