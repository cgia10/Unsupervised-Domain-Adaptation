#https://github.com/corenel/pytorch-adda

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    src_classifier = init_model(net=LeNetClassifier(), restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder(), restore=params.tgt_encoder_restore)

    # eval target encoder on target test set
    print("=== Evaluating classifier for encoded target domain ===")
    print("")
    print("Domain adaption:")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
