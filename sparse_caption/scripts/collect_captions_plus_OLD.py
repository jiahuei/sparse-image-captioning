# -*- coding: utf-8 -*-
"""
Created on 28 Aug 2019 17:15:59

@author: jiahuei
"""
from link_dirs import pjoin
import os
import json
import pickle
import math
import random
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fmng
from PIL import Image, ImageEnhance, ImageOps, ImageFont, ImageDraw

# Variables
SORT_BY_METRIC = 'CIDEr'
JUMP_TO_IDX = 4970
VISUALISE_ATTENTION = True
RADIX_SAMPLE_TIMESTEP = False
RADIX_NUM_TOKENS = 2
MODEL_NAMES = ['sps_80.0', 'sps_97.5']
BASELINE_NAME = 'baseline'
OUTPUT_DIR = '/home/jiahuei/Documents/1_TF_files/radix_v2/compiled_mscoco_test'
IMAGE_DIR = '/master/datasets/mscoco/val2014'
# OUTPUT_DIR = '/home/jiahuei/Documents/1_TF_files/radix_v2/compiled_insta_val'
# IMAGE_DIR = '/master/datasets/insta/images'
JSON_ROOT = '/home/jiahuei/Documents/1_TF_files/radix_v2'

BASELINE_JSON = pjoin(
    JSON_ROOT,
    '/home/jiahuei/Documents/1_TF_files/radix_v2/mscoco_v2/word_w256_LSTM_r512_h1_none_cnnFT_SCST_b7C1.0B0.0/run_01___infer_test_b1_lp0.0___08-11_14-58/captions___113287.json'
)
MODEL_JSON = [
    pjoin(JSON_ROOT,
          '/home/jiahuei/Documents/1_TF_files/radix_v2/mscoco_v2/word_w256_LSTM_r512_h1_none_cnnFT_SCST_b7C1.0B0.0/run_01___infer_test_b1_lp0.0___08-11_14-58/captions___113287.json'),
    pjoin(JSON_ROOT,
          '/home/jiahuei/Documents/1_TF_files/radix_v2/mscoco_v2/word_w256_LSTM_r512_h1_none_cnnFT_SCST_b7C1.0B0.0/run_01___infer_test_b1_lp0.0___08-11_14-58/captions___113287.json')
]

BASELINE_SCORES_JSON = pjoin(
    JSON_ROOT,
    '/home/jiahuei/Documents/1_TF_files/radix_v2/mscoco_v2/word_w256_LSTM_r512_h1_none_cnnFT_SCST_b7C1.0B0.0/run_01___infer_test_b1_lp0.0___08-11_14-58/metric_scores_detailed_113287.json'
)
MODEL_SCORES_JSON = [
    pjoin(JSON_ROOT,
          '/home/jiahuei/Documents/1_TF_files/radix_v2/mscoco_v2/word_w256_LSTM_r512_h1_none_cnnFT_SCST_b7C1.0B0.0/run_01___infer_test_b1_lp0.0___08-11_14-58/metric_scores_detailed_113287.json'),
    pjoin(JSON_ROOT,
          '/home/jiahuei/Documents/1_TF_files/radix_v2/mscoco_v2/word_w256_LSTM_r512_h1_none_cnnFT_SCST_b7C1.0B0.0/run_01___infer_test_b1_lp0.0___08-11_14-58/metric_scores_detailed_113287.json')
]

SHORTLISTED_IMGS = ['COCO_val2014_000000346067.jpg']

# Constants
random.seed(3310)
CATEGORIES = dict(
    x='both_wrong',
    y='both_correct',
    b='baseline_correct',
    m='model_correct',
    a='ambiguous',
)
METRICS = ['CIDEr', 'Bleu_4', 'Bleu_3', 'Bleu_2', 'Bleu_1', 'ROUGE_L', 'METEOR']
IMG_RESIZE = 512
IMG_CROP = int(224 / 256 * IMG_RESIZE)
DISPLAY_BG_SIZE = [int(IMG_RESIZE * 4.5), int(IMG_RESIZE * 3.0)]
BORDER = int(DISPLAY_BG_SIZE[0] / 20)
TEXT_SIZE = int(IMG_RESIZE / 7)
try:
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', TEXT_SIZE)
except OSError:
    FONT_LIST = [f for f in fmng.findSystemFonts(fontpaths=None, fontext='ttf')
                 if 'mono' in os.path.basename(f).lower()]
    font = ImageFont.truetype(FONT_LIST[0], TEXT_SIZE)


def _img_id_to_name(img_id):
    if type(img_id) == str:
        # Insta-1.1M
        img_name = img_id
    else:
        img_name = 'COCO_val2014_{:012d}.jpg'.format(img_id)
    return img_name


def _load_caption_json(res_dict, json_path, name):
    with open(json_path, 'r') as ff:
        captions = json.load(ff)

    pickle_path = json_path.replace("captions___", "outputs___").replace(".json", ".pkl")
    if not os.path.isfile(pickle_path):
        data = None
    else:
        with open(pickle_path, "rb") as ff:
            data = pickle.load(ff)

    for c in captions:
        img_id = c['image_id']
        img_name = _img_id_to_name(img_id)
        if img_id not in res_dict:
            res_dict[img_id] = dict(image_id=img_id, image_name=img_name)
        res_dict[img_id][name] = dict(caption=c['caption'])
        res_dict[img_id][name]['attention'] = data['attention'] if data is not None else None


def _load_score_json(res_dict, json_path, name):
    with open(json_path, 'r') as ff:
        scores = json.load(ff)
    for sc in scores:
        img_id = sc['image_id']
        assert img_id in res_dict
        for m in METRICS:
            res_dict[img_id][name][m] = sc[m]


def _sort_captions(res_dict, sort_metric, sort_model, use_diff=False):
    """
    Return a list of sorted captions.
    :param res_dict: id_to_results
    :param sort_metric: Metric used to sort. If `random`, return list with randomised order.
    :param sort_model: Model result used to sort.
    :param use_diff: If True, use the difference in score between model and baseline to sort.
    :return: A list of sorted captions.
    """
    if isinstance(sort_model, list):
        assert len(sort_model) > 0
    else:
        sort_model = [sort_model]
    res = list(res_dict.values())
    if sort_metric in METRICS:
        def _get_model_mean(elem):
            sc_m = [elem[m][sort_metric] for m in sort_model]
            return sum(sc_m) / len(sc_m)

        if use_diff:
            def _key_fn(elem):
                sc_m = _get_model_mean(elem)
                sc_b = elem[BASELINE_NAME][sort_metric]
                return sc_m - sc_b
        else:
            def _key_fn(elem):
                return _get_model_mean(elem)
        res_sorted = sorted(res, key=_key_fn, reverse=True)
    elif sort_metric == 'random':
        res_sorted = random.shuffle(res)
    else:
        raise ValueError('`sort_metric` must be one of: {}'.format(METRICS + ['random']))
    return res_sorted


def _prepare_img(img_path):
    print(os.path.basename(img_path))
    img = Image.open(img_path)
    img = ImageEnhance.Brightness(img).enhance(1.10)
    img = ImageEnhance.Contrast(img).enhance(1.050)

    # Resize to 512 x 512 instead of 256 x 256
    # Crop to 448 x 448 instead of 224 x 224
    img = img.resize([IMG_RESIZE, IMG_RESIZE], Image.BILINEAR)
    img = ImageOps.crop(img, (IMG_RESIZE - IMG_CROP) / 2)
    return img


def _display_captions(captions_list, sort_metric):
    # Display captions
    print('')
    instructions = [
        '"x" if both are wrong',
        '"y" if both are correct',
        '"b" if baseline is correct',
        '"m" if model is correct',
        '"a" if ambiguous',
        '"e" to exit',
        'other keys to skip.',
        '---\n',
    ]
    instructions = '\n'.join(instructions)
    global JUMP_TO_IDX
    if JUMP_TO_IDX < 0 or JUMP_TO_IDX >= len(captions_list):
        JUMP_TO_IDX = 0

    img_plot = None
    fig = plt.figure(figsize=(20, 10))
    for cap_idx, cap in enumerate(captions_list[JUMP_TO_IDX:]):
        if len(SHORTLISTED_IMGS) > 0 and not any(str(cap['image_id']) in _ for _ in SHORTLISTED_IMGS):
            # Skip if no partial match with any shortlisted images
            continue

        img = _prepare_img(pjoin(IMAGE_DIR, cap['image_name']))

        # Collect info
        base_score = cap[BASELINE_NAME][sort_metric]
        model_score = [cap[n][sort_metric] for n in MODEL_NAMES]
        base_cap = '{} ({:.2f}): {}'.format(
            BASELINE_NAME, base_score, cap[BASELINE_NAME]['caption'])
        model_cap = ['{} ({:.2f}): {}'.format(
            n, model_score[i], cap[n]['caption']) for i, n in enumerate(MODEL_NAMES)]

        # Visualise
        bg_big = Image.new('RGB', DISPLAY_BG_SIZE)
        bg_big.paste(img, (BORDER, int(BORDER * 1.5)))
        draw = ImageDraw.Draw(bg_big)
        draw.text(
            (BORDER, int(BORDER * 0.5)),
            '# {} / {}'.format(JUMP_TO_IDX + cap_idx + 1, len(captions_list)),
            font=font
        )

        # Draw captions
        texts_wrp = []
        for t in [base_cap] + model_cap:
            print(t)
            texts_wrp.append(textwrap.wrap(t, width=45))
        print('')
        offset = int(BORDER * 1.5)
        for text_group in texts_wrp:
            for text in text_group:
                draw.text((BORDER, IMG_RESIZE + offset), text, font=font)
                offset += int(TEXT_SIZE * 1.05)
            offset += TEXT_SIZE

        if img_plot is None:
            img_plot = plt.imshow(bg_big)
        else:
            img_plot.set_data(bg_big)
        plt.show(block=False)
        fig.canvas.draw()

        # Get key press
        # key_input = raw_input(instructions)
        key_input = input(instructions)
        fig.canvas.flush_events()

        if key_input == 'e':
            plt.close()
            break
        elif key_input in CATEGORIES:
            _save_captions(CATEGORIES[key_input], img, cap, bg_big, sort_metric)
        print('')


def _display_attention(captions_list, sort_metric, radix_sample_timestep):
    # Display captions
    print('')
    instructions = [
        '"y" to save',
        '"r" to repeat',
        '"e" to exit',
        'other keys to skip.',
        '---\n',
    ]
    instructions = '\n'.join(instructions)
    global JUMP_TO_IDX
    if JUMP_TO_IDX < 0 or JUMP_TO_IDX >= len(captions_list):
        JUMP_TO_IDX = 0

    model_name = MODEL_NAMES[0]
    img_plot = None
    fig = plt.figure(figsize=(20, 10))
    for cap_idx, cap in enumerate(captions_list[JUMP_TO_IDX:]):
        if len(SHORTLISTED_IMGS) > 0 and not any(str(cap['image_id']) in _ for _ in SHORTLISTED_IMGS):
            # Skip if no partial match with any shortlisted images
            continue

        # Draw attention maps if available (only for the 1st model)
        att_dict = cap[MODEL_NAMES[0]]['attention']
        if att_dict is None:
            continue
        img = _prepare_img(pjoin(IMAGE_DIR, cap['image_name']))

        # Collect info
        model_cap = '{} ({:.2f}): {}'.format(
            model_name, cap[model_name][sort_metric], cap[model_name]['caption']
        )
        sent_len = len(cap[model_name]['caption'].split(' '))

        # Visualise
        bg_big = Image.new('RGB', [IMG_CROP * 6, IMG_CROP * 4])
        bg_big.paste(img, (BORDER, BORDER))
        draw = ImageDraw.Draw(bg_big)
        draw.text(
            (IMG_CROP + BORDER * 2, BORDER),
            '# {} / {}'.format(JUMP_TO_IDX + cap_idx + 1, len(captions_list)),
            font=font
        )
        text_group = textwrap.wrap(model_cap, width=45)
        print(model_cap + '\n')
        for i, text in enumerate(text_group):
            draw.text((IMG_CROP + BORDER * 2, BORDER * 2 + int(TEXT_SIZE * 1.05) * (i + 1)), text, font=font)

        # assert isinstance(att_map_list, list)
        # atts = [_[img_name] for _ in att_map_list]
        # max_len = max(_.shape[1] for _ in atts) + 2
        # bgs = [Image.new('RGB', [IMG_CROP * 6, IMG_CROP * 4]) for _ in range(max_len)]
        # for i, att in enumerate(atts):

        assert isinstance(att_dict, dict)
        att = att_dict[cap['image_id']]
        hw = int(math.sqrt(att.shape[-1]))
        num_heads = att.shape[0]
        att = np.reshape(att, [num_heads, att.shape[1], hw, hw])
        ori_timesteps = att.shape[1]
        if radix_sample_timestep:
            att = att[:, ::RADIX_NUM_TOKENS, :, :]
        sampled_timesteps = att.shape[1]
        att = att[:, :sent_len, :]

        # Apply attention map
        bg = Image.new('RGB', [IMG_CROP, IMG_CROP])
        # border = int(IMG_CROP / 4)
        offset = IMG_CROP + BORDER
        att_comp = [bg_big.copy() for _ in range(att.shape[1] + 1)]

        all_comps = []
        for head in range(num_heads):
            maps = att[head, :, :, :]
            m_max = maps.max()
            # if m_max < 0.01:
            #     maps *= (255.0 / m_max / 5)
            # else:
            #     maps *= (255.0 / m_max)
            maps *= (255.0 / m_max)
            maps = maps.astype(np.uint8)

            comps = []
            for t, m in enumerate(maps):
                m = Image.fromarray(m)
                m = m.convert('L')
                m = m.resize([IMG_CROP, IMG_CROP], Image.BILINEAR)
                comp = Image.composite(img, bg, m)
                comp = ImageEnhance.Brightness(comp).enhance(2.0)
                comp = ImageEnhance.Contrast(comp).enhance(1.5)
                x = (head % 4) * offset + BORDER
                y = int(head / 4) * offset + BORDER * 2 + IMG_CROP
                att_comp[t].paste(comp, (x, y))
                comps.append(comp)
            all_comps.append(comps)

        key_input = 'r'
        while key_input == 'r':
            for comp in att_comp:
                if img_plot is None:
                    img_plot = plt.imshow(comp)
                else:
                    img_plot.set_data(comp)
                plt.show(block=False)
                fig.canvas.draw()
                plt.pause(.05)

            # Get key press
            # key_input = raw_input(instructions)
            key_input = input(instructions)
        fig.canvas.flush_events()

        if key_input == 'e':
            plt.close()
            break
        elif key_input == 'y':
            img_id = cap['image_id']
            score = score = '{:1.3f}'.format(cap[model_name][sort_metric]).replace('.', '-')
            if type(img_id) == str:
                output_dir = pjoin(OUTPUT_DIR, '{}_{}'.format(score, img_id))
            else:
                output_dir = pjoin(OUTPUT_DIR, '{}_{:012d}'.format(score, img_id))
            os.makedirs(output_dir, exist_ok=True)

            img.save(pjoin(output_dir, 'base.jpg'))
            footnote = [
                'Num words (including <EOS>): {}'.format(sent_len + 1),
                'Original attention time steps: {}'.format(ori_timesteps),
                'Sampled time steps before truncation: {}'.format(sampled_timesteps),
                'Sampled time steps after truncation: {}'.format(att.shape[1]),
            ]
            draw.text((BORDER, IMG_CROP + BORDER * 4), '\n\n'.join(footnote), font=font)
            bg_big.save(pjoin(output_dir, 'comp.jpg'))
            with open(pjoin(output_dir, 'caption.txt'), 'w') as f:
                f.write(cap[model_name]['caption'])
            for i, h in enumerate(all_comps):
                for j, t in enumerate(h):
                    if radix_sample_timestep:
                        j *= RADIX_NUM_TOKENS
                    t.save(pjoin(output_dir, 'h{}_t{}.jpg'.format(i, j)))
        print('')


def _save_captions(caption_type, img, caption, composite, sort_metric):
    img_id = caption['image_id']
    base_score = caption[BASELINE_NAME][sort_metric]
    model_score = [caption[n][sort_metric] for n in MODEL_NAMES]
    base_out = '{} ({}): {}'.format(
        BASELINE_NAME, base_score, caption[BASELINE_NAME]['caption'])
    model_out = ['{} ({}): {}'.format(
        n, model_score[i], caption[n]['caption']) for i, n in enumerate(MODEL_NAMES)]
    # Save image
    score = '{:1.3f}'.format(model_score[-1]).replace('.', '-')
    type_short = {v: k for k, v in CATEGORIES.items()}
    if type(img_id) == str:
        img_out_name = '{}_{}_{}.jpg'.format(type_short[caption_type], score, img_id)
    else:
        img_out_name = '{}_{}_{:012d}.jpg'.format(type_short[caption_type], score, img_id)
    img.save(pjoin(OUTPUT_DIR, img_out_name))

    draw = ImageDraw.Draw(composite)
    offset = int(IMG_RESIZE - TEXT_SIZE) / 2
    draw.text((IMG_CROP + BORDER * 2, offset), img_out_name, font=font)
    draw.text((IMG_CROP + BORDER * 2, offset + TEXT_SIZE), 'Type: ' + caption_type, font=font)
    composite.save(pjoin(OUTPUT_DIR, 'comp_' + img_out_name))

    # Write captions
    out_str = '{}\r\n{}\r\n\r\n'.format(base_out, '\r\n'.join(model_out))
    with open(pjoin(OUTPUT_DIR, 'captions_{}.txt'.format(caption_type)), 'a') as f:
        f.write('{}\r\n{}'.format(img_out_name, out_str))

    # Write captions in LATEX format
    modcap = '        \\begin{{modcap}}\n'
    modcap += '            {}\n'
    modcap += '        \\end{{modcap}} \\\\\n'
    out_str = [
        '    \\gph{{1.0}}{{resources/xxx/{}}}  &'.format(img_out_name),
        '    \\begin{tabular}{M{\\linewidth}}',
        '        \\begin{basecap}',
        '            {}'.format(caption[BASELINE_NAME]['caption']),
        '        \\end{basecap} \\\\',
    ]
    for n in MODEL_NAMES:
        out_str += [modcap.format(caption[n]['caption'])]
    out_str += [
        '    \\end{tabular} &',
        '    ',
    ]

    with open(pjoin(OUTPUT_DIR, 'captions_latex_{}.txt'.format(caption_type)), 'a') as f:
        f.write('\n'.join(out_str) + '\n')


def main():
    if len(SHORTLISTED_IMGS) > 0:
        global JUMP_TO_IDX
        JUMP_TO_IDX = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    id_to_results = {}

    config = dict(
        sort_by_metric=SORT_BY_METRIC,
        baseline_json=BASELINE_JSON,
        model_json=MODEL_JSON,
    )
    with open(pjoin(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f)

    # Load captions
    for j, n in zip(MODEL_JSON, MODEL_NAMES):
        _load_caption_json(id_to_results, j, n)
    _load_caption_json(id_to_results, BASELINE_JSON, BASELINE_NAME)

    # Load scores
    for j, n in zip(MODEL_SCORES_JSON, MODEL_NAMES):
        _load_score_json(id_to_results, j, n)
    _load_score_json(id_to_results, BASELINE_SCORES_JSON, BASELINE_NAME)

    # Sort captions
    caption_list = _sort_captions(id_to_results,
                                  sort_metric=SORT_BY_METRIC,
                                  sort_model=MODEL_NAMES,
                                  use_diff=not VISUALISE_ATTENTION)
    if VISUALISE_ATTENTION:
        _display_attention(caption_list, SORT_BY_METRIC, RADIX_SAMPLE_TIMESTEP)
    else:
        _display_captions(caption_list, SORT_BY_METRIC)


if __name__ == '__main__':
    main()
