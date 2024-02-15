import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from sacrebleu.metrics import BLEU, CHRF
from comet import download_model, load_from_checkpoint
import argparse
import numpy as np
from bleurt import score
from io_utils import read_textfile, FILE_PATH, MODELS_DIR

BLEURT_DIR = 'bleurt/BLEURT-20'

def main(args):
    scores = {}

    src, tgt = args.lp.split('-')

    data_path = FILE_PATH.format(args.lp)
    in_data = read_textfile(data_path + 'test.{}'.format(src))
    refs = read_textfile(data_path + 'test.{}'.format(tgt))

    hyp_file_prefix = data_path + '{}/selected_outs/{}_{}'.format(args.model, args.criterion,
                                                                  args.generation)

    assert len(in_data) == len(refs)

    if 'bleu' in args.metrics:
        print("---Computing BLEU scores")
        scores['bleu'] = {}
        bleu = BLEU()

        for n_cands in args.cands_pool:
            print("---number of candidates: {}".format(n_cands))
            hypotheses_file = hyp_file_prefix + '_{}.txt'.format(n_cands)
            hyps = read_textfile(hypotheses_file)
            assert len(in_data) == len(hyps)

            scores['bleu'][n_cands] = bleu.corpus_score(hyps, [refs])

    if 'chrf' in args.metrics:
        print("---Computing chrF scores")
        scores['chrf'] = {}
        chrf = CHRF()

        for n_cands in args.cands_pool:
            print("---number of candidates: {}".format(n_cands))
            hypotheses_file = hyp_file_prefix + '_{}.txt'.format(n_cands)
            hyps = read_textfile(hypotheses_file)
            assert len(in_data) == len(hyps)

            scores['chrf'][n_cands] = chrf.corpus_score(hyps, [refs])

    if 'comet' in args.metrics:
        print("---Computing COMET scores")
        scores['comet'] = {}
        model_path = download_model("Unbabel/wmt22-comet-da", saving_directory=MODELS_DIR)
        model = load_from_checkpoint(model_path)

        for n_cands in args.cands_pool:
            print("---number of candidates: {}".format(n_cands))
            hypotheses_file = hyp_file_prefix + '_{}.txt'.format(n_cands)

            hyps = read_textfile(hypotheses_file)
            assert len(in_data) == len(hyps)

            data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(in_data, hyps, refs)]
            model_output = model.predict(data, batch_size=256, gpus=1)

            scores['comet'][n_cands] = model_output.system_score

        del model
        torch.cuda.empty_cache()

    if 'cometqe' in args.metrics:
        print("---Computing COMET-QE scores")
        scores['comet-qe'] = {}
        model_path = download_model("Unbabel/wmt22-cometkiwi-da", saving_directory=MODELS_DIR)
        model = load_from_checkpoint(model_path)

        for n_cands in args.cands_pool:
            if n_cands == 0:
                hyps = refs
                print("---Computing scores for refs")
            else:
                print("---number of candidates: {}".format(n_cands))
                hypotheses_file = hyp_file_prefix + '_{}.txt'.format(n_cands)

                hyps = read_textfile(hypotheses_file)
                assert len(in_data) == len(hyps)

            data = [{"src": x, "mt": y} for x, y in zip(in_data, hyps)]
            model_output = model.predict(data, batch_size=256, gpus=1)

            scores['comet-qe'][n_cands] = model_output.system_score

        del model
        torch.cuda.empty_cache()

    for metric in scores:
        print("---- {} ----".format(metric))
        for n_cands in scores[metric]:
            print(n_cands, scores[metric][n_cands])

    if 'bleurt' in args.metrics:
        print("---Computing BLEURT scores")
        scores['bleurt'] = {}
        scorer = score.BleurtScorer(BLEURT_DIR)

        for n_cands in args.cands_pool:
            print("---number of candidates: {}".format(n_cands))
            hypotheses_file = hyp_file_prefix + '_{}.txt'.format(n_cands)

            hyps = read_textfile(hypotheses_file)

            bleurt_scores = scorer.score(references=refs, candidates=hyps)

            scores['bleurt'][n_cands] = np.mean(bleurt_scores)

        for n_cands in scores['bleurt']:
            print(n_cands, scores['bleurt'][n_cands])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('score outputs')
    parser.add_argument('--model', type=str,
                        choices=['polylm-1.7b', 'xglm-2.9b', 'llama2-7b', 'llama2-13b', 'mistral',
                                 'alma-7b', 'alma-13b', 'tower', 'nllb-1.3b', 'nllb-3.3b'], help='the model name')
    parser.add_argument('--cands_pool', type=list, default=[5], help='number of candidates')
    parser.add_argument('--metrics', nargs='+', default=['bleu', 'chrf', 'comet', 'bleurt'],
                        choices=['bleu', 'chrf', 'comet', 'cometqe', 'bleurt'], help='the metrics to use')
    parser.add_argument('--criterion', default='cometqe/cometqe-fusion-beam5-kbest0', type=str,
                        help='the criterion used to select outputs')
    parser.add_argument('--generation', default='sample-t0.6',
                        help='the method used to generate candidates')
    parser.add_argument('--lp', help='the language pair')
    args = parser.parse_args()

    main(args)
