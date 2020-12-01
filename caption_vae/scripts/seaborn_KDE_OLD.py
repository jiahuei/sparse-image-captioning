# -*- coding: utf-8 -*-
"""
Created on 21 May 2020 16:56:38
@author: jiahuei
"""

import os
import re
import argparse
import logging
import zipfile
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import graph_util
from matplotlib import pyplot as plt
from scipy.stats import mstats
from copy import deepcopy
from link_dirs import CURR_DIR
from infer_v2 import main as infer_main
from src_com.models import CaptionModel
from common.mask_prune import pruning
from common.configuration_v1 import load_config
from common.utils import configure_logging

logger = logging.getLogger(__name__)
pjoin = os.path.join
FIG_DPI = 600
P_CKPT = re.compile(r'\d+')
# https://stackoverflow.com/a/1176023
P_CAMELCASE = re.compile(r'(?<!^)(?=[A-Z])')
CKPT_PREFIX = 'model_sparse'


# noinspection PyTypeChecker
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--checkpoint_file',
        '-c',
        type=str,
        required=True,
        help='The checkpoint file containing the checkpoint files to convert.')
    parser.add_argument(
        '--save_unmasked_model',
        type=bool,
        default=True,
        help='Boolean. If True, additionally save model without applying mask.')

    parser.add_argument(
        '--infer_on_test',
        type=bool,
        default=False,
        help='Boolean. If True, run the final sparse model on the test set.')
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='The gpu number.')
    parser.add_argument(
        "--logging_level",
        type=int,
        default=20,
        choices=[40, 30, 20, 10],
        help="int: Logging level. ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10.",
    )
    args = parser.parse_args()
    configure_logging(logging_level=args.logging_level)
    return args


def convert_model(config, curr_save_dir, curr_ckpt_path, save_unmasked_model):
    logger.info("Converting `{}`".format(curr_ckpt_path))
    ckpt_dir, ckpt_file = os.path.split(curr_ckpt_path)
    ckpt_num = int(P_CKPT.findall(ckpt_file)[0])  # Checkpoint number

    # Setup input pipeline & Build model
    logger.debug('TensorFlow version: r{}'.format(tf.__version__))
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(config.rand_seed)

        with tf.name_scope('infer'):
            dummy_inputs = (
                tf.zeros(shape=[1] + config.cnn_input_size + [3], dtype=tf.float32),
                # tf.random_uniform(shape=[1] + [448, 448, 3], dtype=tf.float32),
                tf.zeros(shape=[1, 1], dtype=tf.int32),
            )
            # Required for model building
            config.infer_beam_size = 1
            config.infer_length_penalty_weight = 0
            config.infer_max_length = 1
            config.batch_size_infer = 1
            m_infer = CaptionModel(
                config,
                mode='infer',
                batch_ops=dummy_inputs,
                reuse=False,
                name='inference'
            )
            _fake_lstm_forward(config, 49)
        m_outputs = tf.identity(m_infer.infer_output[0], name='output')

        model_vars = tf.contrib.framework.filter_variables(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Model'),
            include_patterns=['Model'],
            exclude_patterns=['mask'],
            reg_search=True)

        pruned_weights = [
            w for w in model_vars
            if 'bias' not in w.op.name and any(_ in w.op.name for _ in ('decoder', 'kernel', 'weights'))
        ]
        logger.info("Visualising {} layers.".format(len(pruned_weights)))
        logger.debug("Layers:\n{}".format(',\n'.join([w.op.name for w in pruned_weights])))
        flat_weights = tf.concat([tf.reshape(w, [-1]) for w in pruned_weights], 0)

        # Apply masks
        masks, _ = pruning.get_masks()
        is_sparse = len(masks) > 0
        if is_sparse:
            assert set(w.op.name for w in pruned_weights) == set(w.op.name for w in pruning.get_weights())
            with tf.name_scope('apply_masks'):
                mask_ops = [tf.multiply(m, w) for m, w in zip(masks, pruned_weights)]
                weight_assign_ops = [tf.assign(w, w_m) for w, w_m in zip(pruned_weights, mask_ops)]

        init_fn = tf.local_variables_initializer()
        restore_saver = tf.train.Saver()
        sparse_saver = tf.train.Saver(var_list=model_vars)

    # Output Naming
    net_name = config.cnn_name.title().replace('Masked_', '')
    net_name = net_name.replace('net', 'Net')
    output_suffix = "{}-{}".format(net_name, config.rnn_name)
    fig_title = ""
    if is_sparse:
        if config.supermask_type == 'regular':
            fig_title = "Proposed, "
        elif config.supermask_type == 'mag_grad_uniform':
            fig_title = "Gradual, "
        elif config.supermask_type == 'mag_blind':
            fig_title = "Hard-blind, "
        elif config.supermask_type == 'mag_uniform':
            fig_title = "Hard-uniform, "
        elif config.supermask_type == 'mag_dist':
            fig_title = "Hard-distribution, "
        else:
            raise ValueError("Invalid pruning type: `{}`".format(config.supermask_type))
        fig_title += "{:.1f}% sparse, ".format(config.supermask_sparsity_target * 100)
        # TexStudio cannot accept filename with dot
        output_suffix += "_{}_{}".format(
            int(config.supermask_sparsity_target * 100),
            ''.join(_.title() for _ in config.supermask_type.split('_'))
        )
    fig_title += "{} + {}".format(net_name.replace('_', '-'), config.rnn_name)
    # TexStudio will annoyingly highlight underscores in filenames
    output_suffix = output_suffix.replace('_', '-')

    # https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-577234513
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/options.md
    profiler_logs_path = pjoin(curr_save_dir, 'profiler_logs_{}.txt'.format(output_suffix))
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    opts['hide_name_regexes'] = ['.*Initializer.*', '.*gen_masks.*', '.*apply_masks.*']
    opts['output'] = 'file:outfile={}'.format(profiler_logs_path)
    with open(profiler_logs_path, 'a') as f:
        flops = tf.profiler.profile(graph=g, run_meta=run_meta, cmd='scope', options=opts)
        f.write('\n\nTotal FLOP count: {}\n'.format(flops.total_float_ops))
    process_profiler_file(profiler_logs_path)

    sess = tf.Session(graph=g)
    with sess:
        sess.run(init_fn)
        restore_saver.restore(sess, curr_ckpt_path)
        g.finalize()

        flat_weights_np = sess.run(flat_weights)
        sps_no_mask = np.sum(flat_weights_np == 0) / flat_weights_np.shape[0] * 100.0

        if is_sparse:
            # Before applying masks
            print('\n==> Model sparsity before applying masks: {:5.1f} %'.format(sps_no_mask))
            if save_unmasked_model:
                _ = sparse_saver.save(
                    sess, pjoin(curr_save_dir, 'model'), ckpt_num, write_meta_graph=False
                )
            # Apply masks
            sess.run(weight_assign_ops)
            # After applying masks
            flat_weights_np = sess.run(flat_weights)
            sps_with_mask = np.sum(flat_weights_np == 0) / flat_weights_np.shape[0] * 100.0
            assert abs(sps_with_mask - (config.supermask_sparsity_target * 100)) < 5.0, (
                "Actual sparsity ({}) differs from target ({}) by more than 5%.".format(
                    sps_with_mask, config.supermask_sparsity_target * 100
                )
            )
            print('==> Model sparsity after applying masks: {:5.1f} %\n'.format(sps_with_mask))
            save_path_full = sparse_saver.save(
                sess, pjoin(curr_save_dir, CKPT_PREFIX), ckpt_num, write_meta_graph=False
            )
            zip_files(curr_save_dir, save_path_full)
        else:
            print('\n==> Dense model sparsity: {:5.1f} %'.format(sps_no_mask))
            save_path_full = sparse_saver.save(
                sess, pjoin(curr_save_dir, 'model'), ckpt_num, write_meta_graph=False
            )
            zip_files(curr_save_dir, save_path_full)

    # pb_path = reload_and_freeze_graph(config=config, save_dir=curr_save_dir, ckpt_path=save_path_full)
    # calculate_flop_from_pb(pb_path)

    # Save non-zero weights for visualisation purposes
    nonzero_weights = flat_weights_np[flat_weights_np != 0]
    np.save(pjoin(curr_save_dir, 'nonzero_weights_flat.npy'), nonzero_weights)

    # Histogram and KDE
    logger.info("Plotting graphs.")
    # plot_kde(
    #     # data=nonzero_weights[np.logical_and(-0.01 <= nonzero_weights, nonzero_weights <= 0.01)],
    #     data=mstats.winsorize(flat_weights_np, limits=0.001),
    #     output_fig_path=pjoin(curr_save_dir, 'KDE-0.01-{}.png'.format(output_suffix.replace('_', '-'))),
    #     fig_title='Distribution of Non-zero Weights in [-0.01, 0.01]\n({})'.format(fig_title),
    #     fig_footnote=None,
    # )
    for i, clip_pct in enumerate([0.005, 0.001]):
        plot_kde(
            data=mstats.winsorize(nonzero_weights, limits=clip_pct),
            # TexStudio will annoyingly highlight underscores in filenames
            output_fig_path=pjoin(curr_save_dir, 'KDE-{}-{}.png'.format(i, output_suffix)),
            fig_title='Distribution of Non-zero Weights\n({})'.format(fig_title),
            fig_footnote='* {:.1f}% winsorization'.format(clip_pct * 100),
        )
    return is_sparse


def process_profiler_file(profiler_file_path):
    assert '_logs_' in profiler_file_path
    with open(profiler_file_path, 'r') as f:
        data = [_.strip() for _ in f.readlines() if _.strip().endswith('flops)')]
    assert data[0].startswith('_TFProfRoot')
    processed = []
    for d in data:
        d = d.replace(' flops)', '').replace(' (', ',')
        d = rreplace(d, '/', ',', 1)
        d = d.split(',')
        assert len(d) == 3, "Saw {}".format(d)
        if d[1] == d[2]:
            d[1] = '-'
        val = None
        if d[-1].endswith('b'):
            val = float(d[-1].replace('b', '')) * 1e9
        elif d[-1].endswith('m'):
            val = float(d[-1].replace('m', '')) * 1e6
        elif d[-1].endswith('k'):
            val = float(d[-1].replace('k', '')) * 1e3
        else:
            pass
        if val:
            d.append(str(int(val)))
        else:
            d.append(d[-1])
        processed.append(','.join(d))
    with open(profiler_file_path.replace('_logs_', '_csv_'), 'w') as f:
        f.write('\n'.join(processed))


def rreplace(s, old, new, occurrence):
    # https://stackoverflow.com/a/2556252
    li = s.rsplit(old, occurrence)
    return new.join(li)


def zip_files(curr_save_dir, save_path_full):
    # TODO: Consider using https://pypi.org/project/zstandard/
    # https://hackernoon.com/when-smallers-better-4b54cedc3402
    sparse_ckpt_files = [
        f for f in os.listdir(curr_save_dir) if os.path.basename(save_path_full) in f
    ]
    logger.info("Packing checkpoint files into a ZIP file.")
    with zipfile.ZipFile(save_path_full + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sparse_ckpt_files:
            zf.write(pjoin(curr_save_dir, f), f)


def plot_kde(data, output_fig_path, fig_title, fig_footnote=None):
    sns.set()
    # print(sns.axes_style())
    sns.set_style(
        "whitegrid", {
            'axes.edgecolor': '.5',
            'grid.color': '.87',
            'grid.linestyle': "dotted",
            # 'lines.dash_capstyle': 'round',
        }
    )
    # colours = ('goldenrod', 'sandybrown', 'chocolate', 'peru')
    # colours = ('c', 'cadetblue', 'lightseagreen', 'skyblue')
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=FIG_DPI, figsize=(8.5, 6.25))
    ax = sns.distplot(
        data,
        bins=50,
        kde_kws={'gridsize': 200, "color": "darkcyan"},
        color='c',
        ax=ax,
    )
    sns.despine()
    # plt.legend(loc='upper left', bbox_to_anchor=(0.1, 1.), fontsize='small')
    plt.title(fig_title)
    if isinstance(fig_footnote, str):
        plt.figtext(
            0.90, 0.025,
            fig_footnote,
            horizontalalignment='right',
            fontsize='xx-small',
        )
    plt.savefig(output_fig_path)
    plt.clf()


def _fake_lstm_forward(config, fm_size=49):
    # attention apply
    with tf.variable_scope('fake_attention_apply'):
        alignments = tf.zeros(shape=[1, 1, 1, fm_size], dtype=tf.float32)
        values = tf.zeros(shape=[1, 1, fm_size, config.attn_size], dtype=tf.float32)
        context = tf.reshape(tf.matmul(alignments, values), [1, config.attn_size])

    with tf.variable_scope('fake_lstm'):
        inputs = tf.zeros(shape=[1, config.rnn_word_size], dtype=tf.float32)
        m_prev = tf.zeros(shape=[1, config.rnn_size], dtype=tf.float32)
        c_prev = tf.zeros(shape=[1, config.rnn_size], dtype=tf.float32)
        kernel = tf.zeros(
            shape=[config.attn_size + config.rnn_word_size + config.rnn_size, config.rnn_size * 4],
            dtype=tf.float32
        )
        bias = tf.zeros(shape=[config.rnn_size * 4], dtype=tf.float32)
        lstm_matrix = tf.matmul(
            tf.concat([inputs, context, m_prev], 1), kernel)
        lstm_matrix = tf.nn.bias_add(lstm_matrix, bias)
        i, j, f, o = tf.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        c = (tf.nn.sigmoid(f + 1.0) * c_prev + tf.nn.sigmoid(i) * tf.nn.tanh(j))
        m = tf.nn.sigmoid(o) * tf.nn.tanh(c)


def reload_and_freeze_graph(config, save_dir, ckpt_path):
    config = deepcopy(config)
    config.supermask_type = None
    config.cnn_name = config.cnn_name.replace('masked_', '')
    config.is_sparse = False  # Treat sparse model as a regular one

    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(config.rand_seed)

        with tf.name_scope('infer'):
            dummy_inputs = (
                tf.zeros(shape=[1] + config.cnn_input_size + [3], dtype=tf.float32),
                tf.zeros(shape=[1, 1], dtype=tf.int32),
            )
            # Required for model building
            config.infer_beam_size = 1
            config.infer_length_penalty_weight = 0
            config.infer_max_length = 2
            config.batch_size_infer = 1
            m_infer = CaptionModel(
                config,
                mode='infer',
                batch_ops=dummy_inputs,
                reuse=False,
                name='inference'
            )
        m_outputs = tf.identity(m_infer.infer_output[0], name='output')
        restore_saver = tf.train.Saver()

    sess = tf.Session(graph=g)
    with sess:
        # Restore model from checkpoint
        restore_saver.restore(sess, ckpt_path)
        g.finalize()
        # https://stackoverflow.com/a/47561171
        # https://stackoverflow.com/a/50680663
        graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), ['output']
        )
    pb_path = pjoin(save_dir, 'graph.pb')
    with tf.gfile.GFile(pb_path, 'wb') as f:
        f.write(graph_def.SerializeToString())
    return pb_path


def calculate_flop_from_pb(pb_path):
    # https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-577234513
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    g = load_pb(pb_path)
    with g.as_default():
        # flops = tf.profiler.profile(g2, options=tf.profiler.ProfileOptionBuilder.float_operation())
        flops = tf.profiler.profile(graph=g, run_meta=run_meta, cmd='scope', options=opts)
        print('FLOP after freezing', flops.total_float_ops)
    pass


def load_pb(pb_path):
    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


def main(args):
    args = deepcopy(args)
    if args.infer_on_test and args.gpu == '':
        raise ValueError('GPU must be used for inference. Specify a GPU ID if `infer_on_test` is True.')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if os.path.isfile(args.checkpoint_file + '.index'):
        ckpt_file = args.checkpoint_file.replace('.index', '')
    elif os.path.isfile(args.checkpoint_file):
        ckpt_file = args.checkpoint_file
    else:
        raise ValueError('`checkpoint_file` must be a file.')

    ckpt_dir = os.path.split(ckpt_file)[0]
    c = load_config(pjoin(ckpt_dir, 'config.pkl'))
    vars(c).update(vars(args))

    # sparse_save_dir = c.log_path = '{}_{}___{}'.format(
    #     ckpt_dir, 'sparse', strftime('%m-%d_%H-%M', localtime()))
    sparse_save_dir = c.log_path = '{}_{}'.format(ckpt_dir, 'sparse')
    if os.path.exists(sparse_save_dir):
        logger.info("Found `{}`. Skipping.".format(sparse_save_dir))
        exit(0)
    os.mkdir(sparse_save_dir)

    is_sparse_model = convert_model(
        config=c,
        curr_save_dir=sparse_save_dir,
        curr_ckpt_path=ckpt_file,
        save_unmasked_model=args.save_unmasked_model
    )

    c.supermask_type = None
    c.cnn_name = c.cnn_name.replace('masked_', '')
    c.is_sparse = False  # Treat sparse model as a regular one
    c.save_config_to_file()

    if args.infer_on_test and is_sparse_model:
        vars(args).update(vars(c))
        args.infer_set = 'test'
        args.infer_checkpoints_dir = sparse_save_dir
        args.infer_checkpoints = 'all'
        args.ckpt_prefix = CKPT_PREFIX
        dataset = c.dataset_file_pattern.split('_')[0]
        if 'coco' in dataset:
            args.annotations_file = 'captions_val2014.json'
        elif 'insta' in dataset:
            args.annotations_file = 'insta_testval_clean.json'
        else:
            raise NotImplementedError('Invalid dataset: {}'.format(dataset))
        args.run_inference = True
        args.get_metric_score = True
        args.save_attention_maps = False
        args.per_process_gpu_memory_fraction = 0.75
        args.infer_beam_size = 3
        args.infer_length_penalty_weight = 0.
        args.infer_max_length = 30
        args.batch_size_infer = 25
        logger.info("Running inference on test set.")
        infer_main(args)
    print('\n')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main(parse_args())
