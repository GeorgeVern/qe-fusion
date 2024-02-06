import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from peft import PeftModel
from io_utils import read_textfile, MODELS_DIR, FILE_PATH
from llm_inference_utils import sample_llm, mt_prompts

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_dict = {'polylm-1.7b': 'DAMO-NLP-MT/polylm-1.7b',
              'xglm-2.9b': 'facebook/xglm-2.9B',
              'llama2-7b': 'meta-llama/Llama-2-7b-hf',
              'llama2-13b': 'meta-llama/Llama-2-13b-hf',
              'mistral': 'mistralai/Mistral-7B-v0.1',
              'alma-7b': 'haoranxu/ALMA-7B-Pretrain',
              'alma-13b': 'haoranxu/ALMA-13B-Pretrain',
              'tower': 'Unbabel/TowerBase-7B-v0.1',
              'nllb-1.3b': 'facebook/nllb-200-1.3B',
              'nllb-3.3b': 'facebook/nllb-200-3.3B',
              }

nllb_lang_codes = {'en': 'eng_Latn', 'de': 'deu_Latn', 'zh': 'zho_Hant', 'fr': 'fra_Latn', 'nl': 'nld_Latn',
                   'is': 'isl_Latn'}


def main(args):
    prompt = mt_prompts[args.lp]

    data_path = FILE_PATH.format(args.lp)
    src_lang, tgt_lang = args.lp.split('-')
    print("Loading data for {}".format(args.lp))
    source_sent = read_textfile(data_path + '{}.{}'.format(args.split, src_lang))

    if 'nllb' not in args.model:
        full_prompt = prompt.build_prompt(args.exemplars)
        print(full_prompt)
        source_data = [full_prompt + "\n" + prompt.template.format(source_sent) for source_sent in source_sent]
    else:
        source_data = source_sent

    if 'nllb' in args.model:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dict[args.model], cache_dir=MODELS_DIR)
    else:
        if 'llama2' in args.model:
            torch_type = torch.bfloat16
        elif 'alma' in args.model:
            torch_type = torch.float16
        else:
            torch_type = 'auto'
        model = AutoModelForCausalLM.from_pretrained(model_dict[args.model], cache_dir=MODELS_DIR, torch_dtype=torch_type)
        if 'alma' in args.model:
            model = PeftModel.from_pretrained(model, model_dict[args.model] + '-LoRA')
    tokenizer = AutoTokenizer.from_pretrained(model_dict[args.model], token="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")

    if 'nllb' in args.model:
        lang_id = tokenizer.lang_code_to_id[nllb_lang_codes[tgt_lang]]
    elif 'llama2' in args.model or 'mistral' in args.model or 'polylm' in args.model:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
    elif 'xglm' in args.model:
        tokenizer.padding_side = "left"
    num_sequences = args.sample if args.decoding_alg == 'sample' else 1

    model = model.to(device)
    output_filename = data_path + '/{}/{}_{}_{}_{}.txt'.format(args.model, args.split, args.model,
                                                               args.decoding_alg + "-t{}".format(
                                                                   args.temperature) if args.decoding_alg == 'sample' else args.decoding_alg,
                                                               num_sequences)
    print("Saving to {}".format(output_filename))

    if args.suffix:
        output_filename = output_filename.replace('.txt', '_{}.txt'.format(args.suffix))

    if args.cont:
        num_lines_written = len(read_textfile(output_filename))
        if num_lines_written == 0:
            raise ValueError("Warning: the number of lines written is 0")
        if num_lines_written % args.bsize != 0:
            raise ValueError("Warning: the number of lines written is not a multiple of the batch size")
        if num_lines_written % num_sequences != 0:
            raise ValueError("Warning: the number of lines written is not a multiple of the number of sequences")
        steps_done = num_lines_written // num_sequences
        print("Continuing from step {}".format(steps_done // args.bsize))
        source_data = source_data[steps_done:]

    print("Sampling from the model with temperature {}".format(args.temperature))

    sample_llm(source_data=source_data, tokenizer=tokenizer, model=model, device=device, batch_size=args.bsize,
               decode_algo=args.decoding_alg, num_sequences=num_sequences,
               gen_filename=output_filename, temperature=args.temperature,
               lang_id=lang_id if 'nllb' in args.model else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query the LLM for a specific task.')
    parser.add_argument('--model', type=str,
                        choices=['polylm-1.7b', 'xglm-2.9b', 'llama2-7b', 'llama2-13b', 'mistral',
                                 'alma-7b', 'alma-13b', 'tower', 'nllb-1.3b', 'nllb-3.3b'],
                        help='the model')
    parser.add_argument('--split', type=str, choices=['train', 'validation', 'test'], help='the split')
    parser.add_argument('--bsize', default=5, type=int, help="the batch size")
    parser.add_argument('--lp', type=str, choices=['en-de', 'zh-en', 'en-ru', 'nl-en', 'de-fr', 'is-en'],
                        help='the language pair')
    parser.add_argument('--decoding_alg', choices=['greedy', 'sample', 'beam'], help='the decoding algorithm')
    parser.add_argument('--sample', type=int, help="the number of samples")
    parser.add_argument('--temperature', type=float, help="the temperature for sampling")
    parser.add_argument('--suffix', default='', type=str, help="add suffix in the output name")
    parser.add_argument('--exemplars', default=8, type=int, help="the number of exemplars for few-shot")
    parser.add_argument('--cont', action='store_true', help="continue from the last step")
    args = parser.parse_args()

    main(args)
