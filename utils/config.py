# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:decoder.py
@time:2018/3/2614:55
@desc:
'''
import argparse
import json

from utils import utils

from enum import Enum


# MODE = Enum('MODE', ('Train', 'Val', 'Test'))
class DATA_PART(Enum):
    Train = 0
    Val = 1
    Test = 2
    Vocab = 3


def datapart_tostr(data_part):
    if data_part == DATA_PART.Train:
        return 'Train'
    elif data_part == DATA_PART.Val:
        return 'Val'
    elif data_part == DATA_PART.Test:
        return 'Test'
    else:
        raise ValueError('Unknown data_part %s' % data_part)


class MODE(Enum):
    Train = 0
    Eval = 1
    Infer = 2


def mode_tostr(mode):
    if mode == MODE.Train:
        return 'Train'
    elif mode == MODE.Eval:
        return 'Eval'
    elif mode == MODE.Infer:
        return 'Infer'
    else:
        raise ValueError('Unknown mode %s' % mode)


def boolean_string(s):
    if type(s) is bool:
        return s
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def merge_configurations(baseline_config_file,
                         model_config_file,
                         argv,
                         parser
                         ):
    if baseline_config_file is not None:
        with open(baseline_config_file, 'r') as fp:
            notes_removed_config_str = utils.remove_json_notes(fp)
            baseline_external_config = json.loads(notes_removed_config_str)
    else:
        baseline_external_config = None

    if model_config_file is not None:
        with open(model_config_file, 'r') as fp:
            notes_removed_config_str = utils.remove_json_notes(fp)
            model_external_config = json.loads(notes_removed_config_str)
            # external_config = json.load(fp)
    else:
        model_external_config = None

    external_config = {}

    if baseline_external_config is not None:
        external_config = baseline_external_config
    if model_external_config is not None:
        for k in model_external_config:
            external_config[k] = model_external_config[k]

    for arg in argv[1:]:
        arg_dest = parser._parse_optional(arg)[0].dest
        if arg_dest in external_config:
            external_config.pop(arg_dest)
    return external_config


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # program config
    group = parser.add_argument_group('Program')
    group.add_argument("--training_manager_function", type=str, default=None,
                       action='store', help="")
    group.add_argument("--config_file", type=str, default=None,
                       help="")
    group.add_argument("--baseline_config_file", type=str, default=None,
                       help="")
    group.add_argument("--random_seed", type=int, default=1234, )
    group.add_argument("--additional", type=str, default=None,
                       help="Additional parameters to be parsed.")
    # network
    group = parser.add_argument_group('Network')
    # initilaize options
    group.add_argument("--initial_func_name", type=str, default="None",
                       choices=["none", "uniform", "standard_uniform", "xavier_uniform", "kaiming", "self_designed"],
                       help="Support uniform and kaiming initialization.")
    group.add_argument("--uniform_maximum", type=float, default=0.1, help="")

    # Embedding options
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument("--text_embedding_size", type=int, default=620, help="The embedding size. default 512D")
    group.add_argument("--share_embedding", type=boolean_string, default=False, )
    group.add_argument("--pretrained_embedding", type=boolean_string, default=False,
                       help="If train the multichannel encoded embedding or not")
    group.add_argument("--embedding_dropout", type=float, default=-1.0)
    group.add_argument("--source_embeddings_path", type=str, default=None, help="")
    group.add_argument("--embedding_word_dict_path", type=str, default=None, help="")
    group.add_argument("--train_text_embeddings_file", type=str, default=None, help="")
    group.add_argument("--val_text_embeddings_file", type=str, default=None, help="")
    group.add_argument("--test_text_embeddings_file", type=str, default=None, help="")

    # Encoder-Decoder options
    group = parser.add_argument_group('Model-Encoder-Decoder')
    group.add_argument("--cell_type", type=str, default='lstm', help="")
    group.add_argument("--bidirectional", type=boolean_string, default=False, help="")
    group.add_argument("--cell_bias", type=boolean_string, default=True,
                       help="Forget bias for BasicLSTMCell.")
    group.add_argument("--dropout", type=float, default=-1.0,
                       help="Dropout rate (not keep_prob), <=0 if not set")
    # Encoder
    group.add_argument("--encoder_num_units", type=int, default=512, help="Network size.")
    group.add_argument("--encoder_num_layers", type=int, default=1, help="")
    group.add_argument("--initialize_encoder_state", type=boolean_string, default=False, help="")
    group.add_argument("--encoder_initial_state_size", type=int, default=2048)
    group.add_argument("--encoder_dropout", type=float, default=-1.0)

    # Attention
    group.add_argument("--attention_type", type=str, default="normed_bahdanau",
                       choices=["bahdanau", "normed_bahdanau",
                                "dot_luong", "general_luong", "luong"])
    group.add_argument("--attention_num_units", type=int, default=512, help="Network size.")
    group.add_argument("--share_attention_parameters", type=boolean_string, default=False,
                       help="Whether to share attention weights among first layer attentions.")
    group.add_argument("--project_out", type=boolean_string, default=False,
                       help="Whether use projected output in attention model or not.")
    group.add_argument("--attention_dropout", type=float, default=0.3)

    # Decoder
    group.add_argument("--decoder_style", type=str, default='bahdanau',
                       choices=['bahdanau', 'luong'])
    group.add_argument("--decoder_num_units", type=int, default=512, help="Network size.")
    group.add_argument("--decoder_num_layers", type=int, default=1, help="")
    group.add_argument("--initialize_decoder_state", type=boolean_string, default=False, help="")
    group.add_argument("--initialize_decoder_with_encoder", type=boolean_string, default=False, help="")
    group.add_argument("--decoder_initial_state_size", type=int, default=2048)
    group.add_argument("--input_feeding", type=boolean_string, default=False)
    group.add_argument("--decoder_output", type=float, default=-1.0)

    # generator
    group.add_argument("--generator_dropout", type=float, default=-1.0)
    group.add_argument("--generator_bias", type=boolean_string, default=False)
    group.add_argument("--share_generator", type=boolean_string, default=False)
    group.add_argument("--project_before_generate", type=boolean_string, default=False)

    # transformer specified parameters
    group = parser.add_argument_group('Model-Transformer')
    group.add_argument("--number_of_head", type=int, default=4)
    group.add_argument("--transformer_activation", type=str, default="relu")

    # imagine transformer
    group = parser.add_argument_group('Model-Imagination-Transformer')
    group.add_argument("--imagine_to_src", type=boolean_string, default=False, )
    group.add_argument("--share_decoder", type=boolean_string, default=False, )
    group.add_argument("--share_optimizer", type=boolean_string, default=False)
    group.add_argument("--imaginate_inference", type=boolean_string, default=False)

    # doubly attentive mmt
    group = parser.add_argument_group("Doubly-Attentive-MMT")
    group.add_argument("--project_multimodal", type=boolean_string, default=False, )
    group.add_argument("--multimodal_output_dim", type=int, default=1024, )
    group.add_argument("--num_multimodal_projector_layer", type=int, default=1, )
    group.add_argument("--multimodal_projector_activation", type=str, default="tanh")
    group.add_argument("--multimodal_projector_bias", type=boolean_string, default=False)
    group.add_argument("--multimodal_projector_dropout", type=float, default=0.0)
    group.add_argument("--big_multimodal_project", type=boolean_string, default=False)
    group.add_argument("--gate_text_attention", type=boolean_string, default=False, help="")
    group.add_argument("--gate_image_attention", type=boolean_string, default=False, help="")
    group.add_argument("--attention_gate_type", type=str, default="scalar",
                       choices=["vector", "scalar"],
                       help="")
    # imagination RNN-Based NMT
    group = parser.add_argument_group("Imagination-RNN_Based-NMT")
    group.add_argument("--image_decoder_activation", type=str, default='tanh')
    group.add_argument("--cosine_similarity_margin", type=float, default=0.1, help="")
    group.add_argument("--project_hidden_after_encode", type=boolean_string, default=False, help="")
    group.add_argument("--project_image_before_loss", type=boolean_string, default=False, help="")

    # MCAE Model parameters
    group = parser.add_argument_group("MCAE-Model")
    group.add_argument("--num_mcae_layers", type=int, default=1)
    group.add_argument("--text_channel_dims", type=int, default=620)
    group.add_argument("--image_channel_dims", type=int, default=2048)
    group.add_argument("--multi_channel_dims", type=int, default=2668)
    group.add_argument("--text2image_activation", type=str, default='tanh')
    group.add_argument("--text2image_dropout", type=float, default=0.0)
    group.add_argument("--mcae_gate_type", type=str, default='scalar',
                       choices=['scalar', 'vector'])
    group.add_argument("--mcae_gate_activation", type=str, default='sigmoid')
    group.add_argument("--mcae_dropout", type=float, default=0.0)
    group.add_argument("--mcae_activation", type=str, default='tanh')
    group.add_argument("--combine_modalities", type=boolean_string, default=False)
    group.add_argument("--text2image_model_path", type=str, default=None)
    group.add_argument("--mcae_model_path", type=str, default=None, help="")

    # MCAEMMT parameters
    group = parser.add_argument_group("MCAEMMT-Model")
    group.add_argument("--use_real_image", type=boolean_string, default=False)
    group.add_argument("--modality_combination_gate_type", type=str, default='vector',
                       choices=['vector', 'scalar'])
    group.add_argument("--modality_combination_gate_activation", type=str, default='sigmoid')

    # transformer entity parameters
    group = parser.add_argument_group("Transformer-Entity")
    group.add_argument("--regressional_text_prediction", type=boolean_string, default=False)
    group.add_argument("--cossim_image_prediction", type=boolean_string, default=False)
    group.add_argument("--token_mask_ratio", type=float, default=0.3)
    group.add_argument("--nmt_train_ratio", type=float, default=1.0, help="")
    group.add_argument("--t2i_train_ratio", type=float, default=0.0, help="")
    group.add_argument("--i2t_train_ratio", type=float, default=0.0, help="")
    group.add_argument("--i2rt_train_ratio", type=float, default=0.0, help="")
    group.add_argument("--a2t_train_ratio", type=float, default=0.0, help="")
    group.add_argument("--a2i_train_ratio", type=float, default=0.0, help="")
    group.add_argument("--project_hidden_to_textual", type=boolean_string, default=False, help="")
    group.add_argument("--project_hidden_to_visual", type=str, default=None,
                       help="linear, self_attention.NUM, autoencoder, or None")
    group.add_argument("--t2i_beta", type=float, default=1.0, help="")
    group.add_argument("--i2t_beta", type=float, default=1.0, help="")
    group.add_argument("--i2rt_beta", type=float, default=1.0, help="")


    # imnoise parameters
    group = parser.add_argument_group("Imnoise")
    group.add_argument("--noise_type", type=str, default=None,
                       choices=['uniform', 'placeholder_entity', 'placeholder_random', 'placeholder_random_fixed', 'fakeimage', 'random_word'])
    group.add_argument("--uniform_noise_dim", type=int, default=2048)
    group.add_argument("--placeholder_id", type=int, default=4)
    group.add_argument("--random_place_ratio", type=float, default=0.2)

    # image projector for all models
    group = parser.add_argument_group("ImageProjectLayer")
    group.add_argument("--big_image_projector", type=boolean_string, default=False,
                       help="a big image projector is stacked like:\n"
                       "\t[[image_dim, image_dim],\n"
                        "\t ......\n"
                       "\t [image_dim, embedding_dim]]"
                       "a small image projector is stacked like:\n"
                       "\t[[image_dim, embedding_dim],\n"
                        "\t ......\n"
                       "\t [embedding_dim, embedding_dim]]")
    group.add_argument("--num_image_project_layer", type=int, default=0, help=">=0")
    group.add_argument("--image_projector_activation", type=str, default=None,
                       choices=[None, 'relu', 'tanh'])
    group.add_argument("--image_projector_bias", type=boolean_string, default=False)
    group.add_argument("--image_projector_dropout", type=float, default=0.0, help="")
    group.add_argument("--image_dropout", type=float, default=0.0)


    # optimizer
    group = parser.add_argument_group('Optimizer')
    group.add_argument("--optimizer", type=str, default="adam",
                       choices=["sgd", "adam", "noamadam"],
                       help="sgd | adam")
    group.add_argument("--adam_beta1", type=float, default=0.9,
                       help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
    group.add_argument("--adam_beta2", type=float, default=0.999,
                       help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
    group.add_argument("--smoothing", type=float, default=0.1, help="for label smoothing")
    group.add_argument("--optimize_delay", type=int, default=1)
    group.add_argument("--max_gradient_norm", type=float, default=1.0,
                       help="Clip gradients to this norm.")
    # scheduler
    group.add_argument("--scheduler_type", type=str, default=None,
                       choices=[None, "StepLR", "ExponentialLR"],
                       help="Scheduler adjusts learning rate."
                            "When noamadam is applied, scheduler_type should be set to None")
    group.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate. Adam: 0.001 | 0.0001")
    group.add_argument("--start_decay_step", type=int, default=0,
                       help="When we start to decay")
    group.add_argument("--decay_step_size", type=int, default=2,
                       help="How often(every N epoch) we decay. For step_size of StepLR.")
    group.add_argument("--decay_factor", type=float, default=0.98,
                       help="How much we decay. For gamma of StepLR and ExponentialLR")

    # Loss criterion
    group = parser.add_argument_group('Criterion')
    group.add_argument("--criterion", type=str, default="CrossEntropyLoss",
                       choices=["CrossEntropyLoss", "LabelSmoothing"],
                       help="supported criterions CrossEntropyLoss|LabelSmoothing.")
    group.add_argument("--loss_normalize_type", type=str, default='token',
                       choices=['token', 'sentence'])

    # Data
    group = parser.add_argument_group('Data')
    # data_manager components config
    group.add_argument("--apply_torchtext", type=boolean_string, default=False, )
    group.add_argument("--iterator", type=str, default='BucketIterator',
                       choices=['BucketIterator', 'OrderedIterator', 'Iterator'])
    group.add_argument("--repeated_iterator", type=boolean_string, default=False,
                       help="if repeat torchtext iterator or not.")
    # dataset options
    group.add_argument("--dataset", type=str, default=None,
                       help='Specify a dataset name "multi30k"|"multi30k_text"|...')
    group.add_argument("--dataset_config_file", type=str, default=None)
    group.add_argument("--extra_dataset", type=str, default=None,
                       help='Specify a dataset name "extra_flickr30k"')
    group.add_argument("--extra_dataset_config_file", type=str, default=None)
    group.add_argument("--src", type=str, default=None,
                       help="Source suffix, e.g., en.")
    group.add_argument("--tgt", type=str, default=None,
                       help="Target suffix, e.g., de.")
    # data process options
    group.add_argument("--bpe_delimiter", type=str, default=None,
                       help="Set to @@ to activate BPE")
    group.add_argument("--shuffle_dataset", type=boolean_string, default=True,
                       help="Only applied to training procedure.")
    group.add_argument("--num_workers", type=int, default=1,
                       help="")
    # Sequence lengths
    group.add_argument("--max_len", type=int, default=50,
                       help="Max length of sequences during training.")
    group.add_argument("--min_len", type=int, default=0,
                       help="Min length of sequences during training.")
    group.add_argument("--src_max_len", type=int, default=50,
                       help="Max length of src sequences during training.")
    group.add_argument("--src_min_len", type=int, default=0,
                       help="Min length of src sequences during training.")
    group.add_argument("--tgt_max_len", type=int, default=50,
                       help="Max length of tgt sequences during training.")
    group.add_argument("--tgt_min_len", type=int, default=0,
                       help="Min length of tgt sequences during training.")

    # images options
    group.add_argument("--image_embedding_size", type=int, default=2048, help="")
    group.add_argument("--global_image_feature_size", type=int, default=2048, help="")
    group.add_argument("--local_image_feature_size", type=int, default=2048, help="")
    group.add_argument("--local_image_feature_width", type=int, default=49, help="")

    # bounding box data options
    group.add_argument("--considered_phrase_type", type=str, default=None, help="")
    group.add_argument("--real_time_parse", type=boolean_string, default=False, help="")
    group.add_argument("--key_word_own_image", type=boolean_string, default=False, help="")
    group.add_argument("--average_multiple_images", type=boolean_string, default=False, help="")

    # Vocab
    group = parser.add_argument_group('Data-Vocab')
    group.add_argument("--no_vocab", type=boolean_string, default=False, help="")
    group.add_argument("--share_vocab", type="bool", nargs="?", const=True,
                       default=False,
                       help="""\
      Whether to use the source vocab and embeddings for both source and
      target.\
      """)
    # special symbols: sos, eos, pad, unk
    group.add_argument("--individual_start_token", type=boolean_string, default=False, )
    group.add_argument("--sos", type=str, default="<sos>",
                       help="Start-of-sentence symbol.")
    group.add_argument("--eos", type=str, default="<eos>",
                       help="End-of-sentence symbol.")
    group.add_argument("--pad", type=str, default="<pad>",
                       help="Pad of a sentence.")
    group.add_argument("--unk", type=str, default="<unk>",
                       help="UNK of a sentence.")
    group.add_argument("--special_tokens", type=str, nargs="+", default=None,
                       help="Special tokens add to the vocabulary, such as <placehoder>, default=None")
    group.add_argument("--max_vocab_size", type=int, default=10000, help="")
    group.add_argument("--load_vocab", type=boolean_string, default=False,
                       help="if load_vocab, load preprocessed vocab 'pt' file, "
                            "--vocab_path should be specified.")
    group.add_argument("--vocab_path", type=str, default=None,
                       help="when load_vocab is True, this should be setup with an existing"
                            "vocabulary path, or this will point to the default path: "
                            "project_path/out_dir/vocab.src-tgt.pt")

    # hardware options
    group = parser.add_argument_group('Hardware')
    group.add_argument("--gpu_id", type=int, default=0)
    group.add_argument("--multiple_gpu", type=boolean_string, default=False,
                       help="")

    # Training options
    group = parser.add_argument_group('Training')
    group.add_argument("--epoch", type=int, default=20,
                       help="The maximum number of epoch to run.")
    group.add_argument("--max_training_steps", type=int, default=1,
                       help="The maximum number of training steps.")
    group.add_argument("--max_steps_without_change", type=int, default=10,
                       help="Used for stoper, the maximum steps without better metrics."
                            "Or the most steps for 'step' as stop_signal.")
    group.add_argument("--steps_per_internal_eval", type=int, default=100,
                       help="The number of step to evalute.")
    group.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    group.add_argument("--batch_first", type=boolean_string, default=False, help="")
    group.add_argument("--stop_signal", type=str, default='loss',
                       help="loss|bleu|step")
    # group.add_argument("--entity_text_train_ratio", type=float, default=0.5)
    # training outputs
    group.add_argument("--out_dir", type=str, default=None,
                       help="Store log/model files.")
    group.add_argument("--premodel_dir", type=str, default=None,
                       help="Pretrained model files.")
    group.add_argument("--project_name", type=str, default=None,
                       help="Store epoch files.")
    group.add_argument("--checkpoint_name", type=str, default="checkpoints",
                       help="Store checkpoint files")
    group.add_argument("--checkpoint_keep_number", type=int, default=1, help="")
    group.add_argument("--averaged_model_checkpoint_name", type=str, default="average_checkpoints")
    group.add_argument("--average_checkpoint", type=boolean_string, default=False, help="")
    group.add_argument("--num_checkpoint_per_average", type=int, default=3, help="")
    group.add_argument("--final_test", type=boolean_string, default=True)
    # multi-task training options
    group.add_argument("--multitask_warmup", type=int, default=0)
    group.add_argument("--multitask_early_stop", type=int, default=-1)

    # Evaluation options
    group = parser.add_argument_group('Evaluation')
    group.add_argument("--eval_batch_size", type=int, default=128, help="Batch size.")
    group.add_argument("--steps_per_external_eval", type=int, default=100,
                       help="The number of step to evalute.")
    group.add_argument("--validate_with_evaluation", type=boolean_string, default=True)
    group.add_argument("--validate_with_inference", type=boolean_string, default=False)
    # Inference options
    group = parser.add_argument_group('Inference')
    group.add_argument("--infer_batch_size", type=int, default=128, help="Batch size.")
    group.add_argument("--steps_per_infer", type=int, default=100,
                       help="")
    group.add_argument("--beam_width", type=int, default=0,
                       help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
    group.add_argument("--max_decode_step_ratio", type=float, default=2.0)
    group.add_argument("--max_decode_step", type=int, default=50,
                       help="The maximum decode steps for inference.")
    group.add_argument("--translation_filename", type=str, default='translations.txt')
    group.add_argument("--metrics", type=str, default="bleu",
                       help=("Comma-separated list of evaluations "
                             "metrics (bleu,rouge,accuracy)"))
    group.add_argument("--bleu_cmd", type=str, default='perl ~/work/tools/multi-bleu.perl',
                       help="")
    group.add_argument("--meteor_cmd", type=str,
                       default='java -Xmx2G -jar {meteor_jar} {hypo} {ref} -l {lang} -norm',
                       help="")
    group.add_argument("--meteor_jar", type=str, default="./utils/meteor-1.5.jar")
    group.add_argument("--infer_record_file", type=str, default=None)
    group.add_argument("--store_attention", type=boolean_string, default=False)

    #
    group.add_argument("--pass_original_embeddings", type=boolean_string, default=False, help="")
    group.add_argument("--if_image_as_word", type=boolean_string, default=False, help="")
    group.add_argument("--project_image_sequence", type=boolean_string, default=False, help="")

    # mutlichannel_autoencoder
    group.add_argument("--train_image_features_file", type=str, default=None, help="")
    group.add_argument("--val_image_features_file", type=str, default=None, help="")
    group.add_argument("--test_image_features_file", type=str, default=None, help="")


    group.add_argument("--if_gate_inputs", type=boolean_string, default=False, help="")
    group.add_argument("--gate_model_type", type=str, default="vector", help="vector or scalar")

    group.add_argument("--train_mca", type=boolean_string, default=False, help="")
    group.add_argument("--finetune_mca", type=boolean_string, default=False, help="")
