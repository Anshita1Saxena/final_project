from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config


def get_args():
    """
    load the configuration files
    :return:
    """
    config = {'dataset_yaml': "./config.yaml",
              'learning_rate': 1e-3,
              'gpuid': -1,
              'seed': 123,
              'init_weight': 0.08,
              'weight_decay': 0,
              'max_epochs': 2,
              'min_freq': 1,
              'grad_clip': 5,
              'batch_size': 80,
              'share_vocab': True,
              'pretrained_word_emb_name': '6B',
              'checkpoint_save_path': "./checkpoint_save",
              'beam_size': 1
              }
    our_args = get_yaml_config(config['dataset_yaml'])
    template = get_basic_args(graph_construction_name=our_args["graph_construction_name"],
                              graph_embedding_name=our_args["graph_embedding_name"],
                              decoder_name=our_args["decoder_name"])
    update_values(to_args=template, from_args_list=[our_args, config])
    return template