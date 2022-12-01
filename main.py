import json
import torch
import logging
import collections
import os.path as osp
# from parser import create_parser

import warnings
warnings.filterwarnings('ignore')

from utils import *
from rfold import RFold


class Exp:
    def __init__(self, args):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()

    def _build_method(self):
        self.method = RFold(self.args, self.device)

    def _get_data(self):
        self.test_loader = get_dataset(self.config)

    def test(self):
        test_f1, test_precision, test_recall, test_runtime = self.method.test_one_epoch(self.test_loader)
        print_log('Test F1: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, Runtime: {3:.4f}\n'.format(test_f1, test_precision, test_recall, test_runtime))
        return test_f1, test_precision, test_recall, test_runtime

# if __name__ == '__main__':
#     RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

#     args = create_parser()
#     config = args.__dict__
#     exp = Exp(args)

#     print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#     exp.test()
#     print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#     test_f1, test_precision, test_recall, test_runtime = exp.test()