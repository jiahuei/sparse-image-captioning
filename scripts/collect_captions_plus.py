# -*- coding: utf-8 -*-
"""
Created on 08 Nov 2020 16:39:46
@author: jiahuei
"""
import os
import pandas as pd
import random
import textwrap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fmng
from typing import Dict, List, Optional
from PIL import Image, ImageEnhance, ImageFont, ImageDraw


class Caption:
    METRICS = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "CIDEr", "METEOR", "ROUGE_L", "SPICE")

    def __init__(self):
        self.df = pd.DataFrame()

    @property
    def data(self):
        return self.df

    def add_data(self, key, caption_json, score_json):
        captions = self._read_json(key, caption_json)
        scores = self._read_json(key, score_json)
        assert len(captions) == len(
            scores
        ), f"Each caption must be paired with its score. Saw {len(captions)} captions and {len(scores)} scores."
        self.df = pd.concat([self.df, captions, scores], axis=1)

    def sort_data(self, metric=None, agg_score="mean", agg_model="mean", use_diff=True):
        if metric is None:
            metric = ["CIDEr"]
        else:
            assert isinstance(metric, list)
        assert agg_score in ("mean", "max", "min")
        assert agg_model in ("mean", "max", "min")

        model_mean = self._agg_scores(
            self.df.loc[:, filter(lambda x: x[0] != "baseline" and x[1] in metric, self.df.columns)],
            agg_score,
            agg_model,
        )
        if use_diff:
            assert "baseline" in self.df.columns
            baseline_mean = self._agg_scores(
                self.df.loc[:, filter(lambda x: x[0] == "baseline" and x[1] in metric, self.df.columns)],
                agg_score,
                agg_model,
            )
            sort_by = model_mean - baseline_mean
        else:
            sort_by = model_mean
        # `sort_by` Series has `image_id` as index
        self.df = self.df.loc[sort_by.sort_values(ascending=False).index]

    @staticmethod
    def _agg_scores(df, agg_score, agg_model):
        # MultiIndex: level 0 = model, level 1 = metric
        return df.groupby(level=0, axis=1).agg(agg_score).agg(agg_model, axis=1)

    @staticmethod
    def _read_json(key, json_filepath):
        assert os.path.isfile(json_filepath), f"Provided path is not a file: `{json_filepath}`"
        df = pd.read_json(json_filepath)
        if "SPICE" in df.columns:
            df.loc[:, "SPICE"] = df.loc[:, "SPICE"].map(lambda x: x["All"]["f"])
        df = df.set_index("image_id")
        df.columns = pd.MultiIndex.from_product([[key], df.columns])
        # df.columns = [f"{key}/{_}" for _ in df.columns]
        return df


class CaptionCollector:
    # Constants
    SEED = 3310
    CATEGORIES = dict(
        x="both_wrong",
        y="both_correct",
        b="baseline_correct",
        m="model_correct",
        a="ambiguous",
    )
    METRICS = ["CIDEr", "Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "ROUGE_L", "METEOR"]
    IMG_RESIZE = 512
    IMG_CROP = int(224 / 256 * IMG_RESIZE)
    DISPLAY_BG_SIZE = [int(IMG_RESIZE * 4.5), int(IMG_RESIZE * 3.0)]
    BORDER = int(DISPLAY_BG_SIZE[0] / 20)
    TEXT_SIZE = int(IMG_RESIZE / 7)
    try:
        FONT = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", TEXT_SIZE)
    except OSError:
        font_list = [
            f
            for f in fmng.findSystemFonts(fontpaths=None, fontext="ttf")
            if "arial" in os.path.basename(f).lower() or "mono" in os.path.basename(f).lower()
        ]
        FONT = ImageFont.truetype(font_list[0], TEXT_SIZE)

    def __init__(
        self,
        image_dir: str,
        baseline_results: Dict,
        model_results: List,
        sort_metric: Optional[List] = None,
        use_diff: Optional[bool] = True,
    ):
        random.seed(self.SEED)
        self.image_dir = image_dir
        self.captions = Caption()
        self.captions.add_data(baseline_results["name"], baseline_results["caption"], baseline_results["score"])
        self.baseline_name = baseline_results["name"]
        self.model_name = []
        for res in model_results:
            self.captions.add_data(res["name"], res["caption"], res["score"])
            self.model_name.append(res["name"])
        self.captions.sort_data(metric=sort_metric, use_diff=use_diff)

    def display_captions(self, output_dir, view_metric="CIDEr", jump_to_idx=0, img_shortlist=None):
        # Display captions
        print("")
        instructions = [
            "'x' if both are wrong",
            "'y' if both are correct",
            "'b' if baseline is correct",
            "'m' if model is correct",
            "'a' if ambiguous",
            "'e' to exit",
            "other keys to skip.",
            "---\n",
        ]
        instructions = "\n".join(instructions)
        if jump_to_idx < 0 or jump_to_idx >= len(self.captions.data):
            jump_to_idx = 0
        if isinstance(img_shortlist, (tuple, list, set)):
            img_shortlist = set(self._img_id_to_name(_) for _ in img_shortlist)
        else:
            img_shortlist = set()

        img_plot = None
        fig = plt.figure(figsize=(20, 10))
        df = self.captions.data.iloc[jump_to_idx:]
        for cap_idx in range(len(df)):
            row = df.iloc[cap_idx]
            image_name = self._img_id_to_name(row.name)
            if len(img_shortlist) > 0 and image_name not in img_shortlist:
                # Skip if no partial match with any shortlisted images
                continue

            img = self._read_img(os.path.join(self.image_dir, image_name))

            # Visualise
            bg_big = Image.new("RGB", self.DISPLAY_BG_SIZE)
            bg_big.paste(img, (self.BORDER, int(self.BORDER * 1.5)))
            draw = ImageDraw.Draw(bg_big)
            draw.text(
                (self.BORDER, int(self.BORDER * 0.5)),
                "# {} / {}".format(jump_to_idx + cap_idx + 1, len(self.captions.data)),
                font=self.FONT,
            )

            # Draw captions
            base_cap = (
                f"{self.baseline_name} ({row[self.baseline_name, view_metric]:.2f}): "
                f"{row[self.baseline_name, 'caption']}"
            )
            model_cap = [f"{_} ({row[_, view_metric]:.2f}): {row[_, 'caption']}" for _ in self.model_name]
            texts_wrp = []
            for t in [base_cap] + model_cap:
                print(t)
                texts_wrp.append(textwrap.wrap(t, width=45))
            print("")
            offset = int(self.BORDER * 1.5)
            for text_group in texts_wrp:
                for text in text_group:
                    draw.text((self.BORDER, self.IMG_RESIZE + offset), text, font=self.FONT)
                    offset += int(self.TEXT_SIZE * 1.05)
                offset += self.TEXT_SIZE

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

            if key_input == "e":
                plt.close()
                break
            elif key_input in self.CATEGORIES:
                self._save_results(self.CATEGORIES[key_input], bg_big, img, row, view_metric, output_dir)
            print("")

    def _save_results(self, caption_type, composite, img, row, view_metric, output_dir):
        img_id = row.name
        base_out = (
            f"{self.baseline_name} ({row[self.baseline_name, view_metric]:.2f}): "
            f"{row[self.baseline_name, 'caption']}"
        )
        model_out = [f"{_} ({row[_, view_metric]:.2f}): {row[_, 'caption']}" for _ in self.model_name]
        # Save image
        score = f"{row[self.model_name[-1], view_metric]:.3f}".replace(".", "-")
        type_short = {v: k for k, v in self.CATEGORIES.items()}
        if isinstance(img_id, str):
            img_out_name = f"{type_short[caption_type]}_{score}_{img_id}.jpg"
        else:
            img_out_name = f"{type_short[caption_type]}_{score}_{img_id:012d}.jpg"
        os.makedirs(output_dir, exist_ok=True)
        img.save(os.path.join(output_dir, img_out_name))

        draw = ImageDraw.Draw(composite)
        offset = int(self.IMG_RESIZE - self.TEXT_SIZE) / 2
        draw.text((self.IMG_CROP + self.BORDER * 2, offset), img_out_name, font=self.FONT)
        draw.text((self.IMG_CROP + self.BORDER * 2, offset + self.TEXT_SIZE), "Type: " + caption_type, font=self.FONT)
        composite.save(os.path.join(output_dir, "comp_" + img_out_name))

        # Write captions
        out_str = "{}\r\n{}\r\n\r\n".format(base_out, "\r\n".join(model_out))
        with open(os.path.join(output_dir, f"captions_{caption_type}.txt"), "a") as f:
            f.write(f"{img_out_name}\r\n{out_str}")

        # Write captions in LATEX format
        modcap = "        \\begin{{modcap}}\n"
        modcap += "            {}\n"
        modcap += "        \\end{{modcap}} \\\\"
        out_str = [
            f"    \\gph{{1.0}}{{resources/xxx/{img_out_name}}}  &",
            "    \\begin{tabular}{M{\\linewidth}}",
            "        \\begin{basecap}",
            f"            {row[self.baseline_name, 'caption']}",
            "        \\end{basecap} \\\\",
        ]
        for n in self.model_name:
            out_str += [modcap.format(row[n, "caption"])]
        out_str += [
            "    \\end{tabular} &",
            "    ",
        ]

        with open(os.path.join(output_dir, f"captions_latex_{caption_type}.txt"), "a") as f:
            f.write("\n".join(out_str) + "\n")

    @staticmethod
    def _img_id_to_name(img_id):
        if isinstance(img_id, str):
            # Insta-1.1M
            img_name = img_id
        else:
            img_name = f"COCO_val2014_{int(img_id):012d}.jpg"
        return img_name

    def _read_img(self, img_path):
        img = Image.open(img_path)
        img = ImageEnhance.Brightness(img).enhance(1.10)
        img = ImageEnhance.Contrast(img).enhance(1.050)

        # # Resize to 512 x 512 instead of 256 x 256
        # # Crop to 448 x 448 instead of 224 x 224
        # img = img.resize([self.IMG_RESIZE, self.IMG_RESIZE], Image.BILINEAR)
        # img = ImageOps.crop(img, (self.IMG_RESIZE - self.IMG_CROP) / 2)
        img = img.resize([self.IMG_CROP, self.IMG_CROP], Image.BILINEAR)
        return img


if __name__ == "__main__":
    LOG_DIR = "/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1"
    collector = CaptionCollector(
        sort_metric=["Bleu_4", "CIDEr"],
        use_diff=True,
        image_dir="/master/datasets/mscoco/val2014",
        baseline_results={
            "name": "baseline",
            "caption": "/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1/UpDownLSTM__baseline/val_beam_5/caption_00108000.json",
            "score": "/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1/UpDownLSTM__baseline/val_beam_5/score_00108000_detailed.json",
        },
        model_results=[
            {
                "name": "UD_99_f16",
                "caption": "/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1/UpDownLSTM__supermask__0.991__SCST_sample_baseline_s60_e10_C1B0001/val_beam_5_float16/caption_00222000.json",
                "score": "/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1/UpDownLSTM__supermask__0.991__SCST_sample_baseline_s60_e10_C1B0001/val_beam_5_float16/score_00222000_detailed.json",
            },
            {
                "name": "ORT_99_f16",
                "caption": "/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1/RTrans__supermask__0.991__SCST_sample_baseline_s15_e10_C1B0001/val_beam_5_float16/caption_00226570.json",
                "score": "/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco_v1/RTrans__supermask__0.991__SCST_sample_baseline_s15_e10_C1B0001/val_beam_5_float16/score_00226570_detailed.json",
            },
        ],
    )
    collector.display_captions(
        output_dir="/home/jiahuei/Documents/1_TF_files/relation_trans/mscoco__compiled_captions",
        img_shortlist=[
            377881,
            72776,
            92202,
            408863,
            327401,
            # 553057,
            # 117563,
            # 340386,
            # 17882,
            # 510877,
            # 455345,
            # 202298,
            # 189932,
            # 428278,
            # 483159,
        ],
        jump_to_idx=4800,
    )
