import argparse
import difflib
from collections import OrderedDict
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
from io_utils import read_textfile, MODELS_DIR, FILE_PATH, cont_write_textfile
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
import numpy as np

BATCH_SIZE = 350
COMET_MODEL = "wmt22-comet-da"
COMETQE_MODEL = "wmt22-cometkiwi-da"

# the window size for the diff computation
WINDOW_SIZE = 1


def compute_diffs(base_sent, alter_sents, ws=1):
    """
    Function to compute the differing spans between the base sentence and the other candidates
    :param base_sent: base sentence
    :param alter_sents: list of candidate sentences
    """
    edits = OrderedDict()
    edit_positions = OrderedDict()
    base_hyp = base_sent.split(" ")
    for candidate in alter_sents:
        alter_hyp = candidate.split(" ")
        # compute the differing spans between the base sentence and the candidate
        s = difflib.SequenceMatcher(None, base_hyp, alter_hyp)
        for op, s_start_ind, s_end_ind, t_start_ind, t_end_ind in s.get_opcodes():
            if op != "equal":
                # if op != "replace":
                s_start_ind -= ws
                t_start_ind -= ws
                s_end_ind += ws
                t_end_ind += ws
                source_span = " ".join(base_hyp[max(s_start_ind, 0): s_end_ind])
                target_span = " ".join(alter_hyp[max(t_start_ind, 0): t_end_ind])
                # add the differing spans to the edits dictionary
                if source_span not in edits:
                    edits[source_span] = [target_span]
                    # save the start index of the initial span to the edit positions dictionary
                    edit_positions[source_span] = [max(s_start_ind - 1, 0)]
                else:
                    if target_span not in edits[source_span]:
                        edits[source_span].append(target_span)
    # sort the edits based on the edit positions
    sorted_positions = OrderedDict(sorted(edit_positions.items(), key=lambda item: item[1]))
    # sorted_edits = OrderedDict((key, edits[key]) for key in sorted_positions)
    sorted_edits = [(key, edits[key]) for key in sorted_positions]
    return sorted_edits


def combine_hyps(outputs, refs, in_data, beam=5, metric='bleu', keep_kbest=0, ws=1):
    """
    Function to combine the hypotheses using pseudo-beam search
    :param outputs: list of lists of candidate outputs
    :param refs: list of references
    :param in_data: list of input sentences
    :param beam: beam size
    :param metric: metric to use for ranking the candidates
    :param keep_kbest: number of best candidates to keep, 0 equals to keeping all candidates
    """
    if metric == 'bleu':
        bleu = BLEU()
    elif metric == 'comet':
        model_path = download_model("Unbabel/{}".format(COMET_MODEL), saving_directory=MODELS_DIR)
        model = load_from_checkpoint(model_path)
    elif metric == 'cometqe':
        model_path = download_model("Unbabel/{}".format(COMETQE_MODEL), saving_directory=MODELS_DIR)
        model = load_from_checkpoint(model_path)
    else:
        raise ValueError

    # cache scores for hypotheses already formed in previous steps
    computed_scores = {id: {} for id in range(len(outputs))}
    # select the k best candidates based on the metric scores
    if keep_kbest == -1:
        pass
    elif keep_kbest < -1:
        outputs = [x[keep_kbest:] for x in outputs]
    else:
        outputs, scores = find_best_hyps(outputs, refs, in_data, metric=metric, k_best=keep_kbest)
    for id in range(len(outputs)):
        for cand, score in zip(outputs[id], scores[id]):
            computed_scores[id][cand] = score

    # Create a list of all differing spans for all references
    all_diffs = [compute_diffs(hyps[0], hyps[1:], ws=ws) if len(hyps) > 1 else {} for hyps in outputs]

    # Use the first candidate as the initial hypothesis
    all_hyps = OrderedDict({sent_id: hyps[:1] for sent_id, hyps in enumerate(outputs)})
    final_outputs = {}
    # Iterate over the number of combinations
    for combination_step in tqdm(range(max(map(len, all_diffs)))):
        hyps_per_input = {}
        # Iterate over all references
        for idx, ref in enumerate(refs):
            if combination_step >= len(all_diffs[idx]):
                if idx not in final_outputs:
                    final_outputs[idx] = all_hyps[idx][
                        0]  # If there are no more diffs, keep the current best hypothesis
                    all_hyps.pop(idx)
                    computed_scores.pop(idx)
                continue

            init_span, alter_spans = all_diffs[idx][combination_step]  # Get the diffs for the current span

            # iterate over the current beam hypotheses
            alter_hyps = []
            for curr_hyp in all_hyps[idx]:
                for span in alter_spans:
                    # create alternative hypotheses by replacing the differing span
                    new_hyp = curr_hyp.replace(init_span, span)
                    # add the alternative hypotheses to the beam search if it is not already there
                    if new_hyp not in all_hyps[idx]:
                        alter_hyps.append(new_hyp)
            all_hyps[idx] += alter_hyps

        flattened_hyps, flattened_inputs, flattened_refs = [], [], []
        for j, beam_hyps in all_hyps.items():
            hyps_per_input[j] = 0
            for hyp in beam_hyps:
                # compute the metric score for the hypothesis if it has not been computed in previous steps
                if hyp not in computed_scores[j]:
                    flattened_hyps.append(hyp)
                    flattened_inputs.append(in_data[j])
                    flattened_refs.append(refs[j])
                    # save the number of novel hypotheses per input sentence
                    hyps_per_input[j] += 1
        # check if there are any hypotheses to score
        if len(flattened_hyps) > 0:
            # compute the metric scores for the beam hypotheses
            if metric == 'bleu':
                beam_metric_scores = [bleu.corpus_score([x], [[y]]).score for x, y in
                                      zip(flattened_hyps, flattened_refs)]
            else:
                if metric == 'comet':
                    data = [{"src": x, "mt": y, "ref": z} for x, y, z in
                            zip(flattened_inputs, flattened_hyps, flattened_refs)]
                else:
                    data = [{"src": x, "mt": y} for x, y in zip(flattened_inputs, flattened_hyps)]
                beam_metric_scores = model.predict(data, batch_size=min(BATCH_SIZE, len(data)), gpus=1,
                                                   progress_bar=False).scores
        else:
            beam_metric_scores = []

        # Regroup the flattened list into a list of lists
        start = 0
        for k in all_hyps:
            end = start + hyps_per_input[k]
            new_hyp_scores = beam_metric_scores[start:end]
            new_hyps = flattened_hyps[start:end]

            # Combine the hypotheses with their scores and update the computed scores dictionary
            hyps_with_scores = list(zip(new_hyps, new_hyp_scores))
            for hyp in all_hyps[k]:
                if hyp not in computed_scores[k]:
                    computed_scores[k][hyp] = new_hyp_scores[new_hyps.index(hyp)]
                else:
                    hyps_with_scores.append((hyp, computed_scores[k][hyp]))

            # Sort the hypotheses by score and keep the top beams
            all_hyps[k] = [x[0] for x in sorted(hyps_with_scores, key=lambda x: x[1], reverse=True)[:beam]]

            start = end

    # Select the best hypothesis for each input sentence
    for idx, beam_hyps in all_hyps.items():
        final_outputs[idx] = beam_hyps[0]

    return [final_outputs[key] for key in sorted(final_outputs.keys())]


def find_best_hyps(outputs, refs, input_data, metric='bleu', k_best=1):
    """
    Function to find the best candidate hypotheses for each input sentence
    :param outputs: list of lists of candidate outputs
    :param refs: list of reference outputs
    :param input_data: list of input sentences
    :param metric: metric to use for ranking the candidates
    :param k_best: number of best candidates to keep, 0 equals to keeping all candidates
    """
    best_outputs, best_scores = [], []
    metric_scores = []
    unique_outputs, unique_inputs, unique_refs = [], [], []
    num_unique_hyps = []
    # remove duplicate hypotheses
    for i, candidates in enumerate(outputs):
        unique_hyps = []
        for j, hyp in enumerate(candidates):
            if hyp not in unique_hyps:
                unique_hyps.append(hyp)
                unique_outputs.append(hyp)
                unique_inputs.append(input_data[i])
                unique_refs.append(refs[i])
        num_unique_hyps.append(len(unique_hyps))

    if metric == 'bleu':
        bleu = BLEU()
        metric_scores += [bleu.corpus_score([hyp], [[ref]]).score for hyp, ref in zip(unique_outputs, unique_refs)]
    else:
        # prepare the data for the automatic metric model
        if metric == 'comet':
            model_path = download_model("Unbabel/{}".format(COMET_MODEL), saving_directory=MODELS_DIR)
            model = load_from_checkpoint(model_path)
            data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(unique_inputs, unique_outputs, unique_refs)]
        elif metric == 'cometqe':
            model_path = download_model("Unbabel/{}".format(COMETQE_MODEL), saving_directory=MODELS_DIR)
            model = load_from_checkpoint(model_path)
            data = [{"src": x, "mt": y} for x, y in zip(unique_inputs, unique_outputs)]
        else:
            raise ValueError
        metric_scores = model.predict(data, batch_size=BATCH_SIZE, gpus=1).scores
    # select the best candidates based on the scores
    start = 0
    for i in range(len(outputs)):
        end = start + num_unique_hyps[i]
        scores_pool = metric_scores[start:end]
        cand_pool = unique_outputs[start:end]
        sorted_lists = sorted(zip(scores_pool, cand_pool), reverse=True)
        sorted_scores, sorted_hyps = map(list, zip(*sorted_lists))
        if k_best == 0:
            best_outputs.append(sorted_hyps)
            best_scores.append(sorted_scores)
        elif k_best > 0 and k_best < 1:
            # filter out the candidates with scores lower than k_best
            filtered_scores, filtered_hyps = [], []
            for i, x in enumerate(sorted_scores):
                # the best candidate is kept even if its score is lower than k_best
                if i == 0:
                    filtered_scores.append(x)
                    filtered_hyps.append(sorted_hyps[i])
                else:
                    if x > k_best:
                        filtered_scores.append(x)
                        filtered_hyps.append(sorted_hyps[i])
            best_outputs.append(filtered_hyps)
            best_scores.append(filtered_scores)
        elif k_best == 1:
            best_outputs.append(sorted_hyps[0])
            best_scores.append(sorted_scores[0])
        else:
            best_outputs.append(sorted_hyps[:k_best])
            best_scores.append(sorted_scores[:k_best])

        start = end

    return best_outputs, best_scores


def mbr(outputs, input_data, metric='comet'):
    """
    Function to compute the Minimum Bayes Risk hypothesis
    :param outputs: list of lists of candidate outputs
    :param input_data: list of input sentences
    :param metric: metric to use for ranking the candidates
    """
    mbr_outputs = []
    mbr_inputs, mbr_hyps, mbr_refs = [], [], []
    num_candidates = len(outputs[0])
    unique_combinations = {}
    # prepare the data for the automatic metric model
    for i, candidates in enumerate(outputs):
        for j1 in range(num_candidates):
            for j2 in range(num_candidates):
                if j1 != j2:
                    combination = (input_data[i], candidates[j1], candidates[j2])

                    if combination not in unique_combinations:
                        mbr_inputs.append(input_data[i])
                        mbr_hyps.append(candidates[j1])
                        mbr_refs.append(candidates[j2])
                        unique_combinations[combination] = len(mbr_inputs) - 1  # Store the index

    # check that the number of inputs, hypotheses and references is correct
    assert len(mbr_inputs) == len(mbr_hyps) == len(mbr_refs)

    metric_scores = []

    # Obtain scores for the n(n-1) pairs of candidates
    if metric == 'bleu':
        bleu = BLEU()
        for cand1, cand2 in zip(mbr_hyps, mbr_refs):
            metric_scores.append(bleu.corpus_score([cand1], [[cand2]]).score)
    elif metric == 'comet':
        model_path = download_model("Unbabel/{}".format(COMET_MODEL), saving_directory=MODELS_DIR)
        model = load_from_checkpoint(model_path)
        data = [{"src": x, "mt": y, "ref": z} for x, y, z in zip(mbr_inputs, mbr_hyps, mbr_refs)]
        metric_scores = model.predict(data, batch_size=BATCH_SIZE, gpus=1).scores
    else:
        raise ValueError

    for i, candidates in enumerate(outputs):
        # Reshape metric_scores into a 2D matrix for unique combinations
        utility_scores = np.zeros((num_candidates, num_candidates))
        for j1 in range(num_candidates):
            for j2 in range(num_candidates):
                if j1 != j2:
                    index = unique_combinations[(input_data[i], candidates[j1], candidates[j2])]
                    utility_scores[j1, j2] = metric_scores[index]

        # Compute candidate with the maximum score
        sum_scores = np.sum(utility_scores, axis=1)
        index = np.argmax(sum_scores)
        mbr_outputs.append(candidates[index])

    return mbr_outputs


def main(args):
    src, tgt = args.lp.split('-')
    data_path = FILE_PATH.format(args.lp)

    in_data = read_textfile(data_path + "test.{}".format(src))
    refs = read_textfile(data_path + "test.{}".format(tgt))

    bleu = BLEU()

    for cand_number in args.cands_pool:
        cand_filename = data_path + "{}/test_{}_{}_{}.txt".format(args.model, args.model, args.generation,
                                                                  cand_number)
        with open(cand_filename, "r") as f:
            textfile = [line.strip() for line in f.readlines()]
        detok_outputs = np.array(textfile).reshape((-1, cand_number)).tolist()

        print("----Selecting outputs from {} using {}-{}-----".format(cand_filename, args.criterion, args.method))
        assert len(detok_outputs) == len(in_data) == len(refs)

        exp_name = '{}/{}'.format(args.criterion, args.criterion)
        if args.criterion == 'modelprob':
            sys = [x[0] for x in detok_outputs]
        else:
            if args.method == 'rank':

                sys, _ = find_best_hyps(detok_outputs, refs, in_data, metric=args.criterion)

                exp_name += '-rank'
            elif args.method == 'mbr':
                sys = mbr(detok_outputs, in_data, metric=args.criterion)

                exp_name += '-mbr'
            else:
                sys = combine_hyps(detok_outputs, refs, in_data, beam=args.beam, metric=args.criterion,
                                   keep_kbest=args.kbest, ws=WINDOW_SIZE)

                exp_name += '-fusion-beam{}-kbest{}'.format(args.beam, args.kbest)

            if args.criterion == 'cometqe' and COMETQE_MODEL != "wmt22-cometkiwi-da":
                exp_name = exp_name.replace(args.criterion, COMETQE_MODEL)

            if WINDOW_SIZE > 1:
                exp_name = exp_name + "-ws{}".format(WINDOW_SIZE)

        output_filename = data_path + "{}/selected_outs/{}_{}_{}.txt".format(args.model, exp_name, args.generation,
                                                                             cand_number)
        cont_write_textfile(sys, output_filename)

        print(bleu.corpus_score(sys, [refs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('score outputs')
    parser.add_argument('--model', type=str,
                        choices=['polylm-1.7b', 'xglm-2.9b', 'llama2-7b', 'llama2-13b', 'mistral',
                                 'alma-7b', 'alma-13b', 'tower', 'nllb-1.3b', 'nllb-3.3b'], help='the model name')
    parser.add_argument('--cands_pool', type=int, nargs='*', default=[5], help='number of candidates')
    parser.add_argument('--criterion', choices=['modelprob', 'bleu', 'comet', 'cometqe'], default='comet',
                        help='the criterion used to select outputs')
    parser.add_argument('--method', choices=['rank', 'mbr', 'fusion'],
                        help='the method used to produce the final output')
    parser.add_argument('--generation', default='sample-t0.6',
                        help='the method used to generate candidates')
    parser.add_argument('--beam', default=5, type=int, help='beam size')
    parser.add_argument('--kbest', default=0, type=float, help='consider top k candidates')
    parser.add_argument('--lp', help='the language pair')
    args = parser.parse_args()
    main(args)
