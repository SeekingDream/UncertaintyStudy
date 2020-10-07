import torch.nn as nn
from BasicalClass.common_function import *
from BasicalClass.BasicModule import BasicModule
import torch.optim as optim
from java_dataset.se_tasks.code_summary.scripts.Code2VecModule import Code2Vec
from java_dataset.se_tasks.code_summary.scripts.CodeLoader import CodeLoader
from java_dataset.se_tasks.code_summary.scripts.main import perpare_train, my_collate
from java_dataset.checkpoint import Checkpoint

class CodeSummary_Module(BasicModule):
    def __init__(self, device, embed_type=1, load_poor=False):
        super(CodeSummary_Module, self).__init__(device, load_poor)

        self.embed_type = embed_type
        self.tk_path = 'java_dataset/data/java-small-preprocess/tk.pkl'
        self.train_path = 'java_dataset/data/java-small-preprocess/train.pkl'
        self.test_path = 'java_dataset/data/java-small-preprocess/test.pkl'
        self.val_path = 'java_dataset/data/java-small-preprocess/val.pkl'
        self.vec_path = 'java_dataset/embedding_vec/100_2/Doc2VecEmbedding0.vec'
        self.embed_dim = 100
        self.out_dir = 'java_dataset/se_tasks/code_summary/result'
        # self.res_dir = 'java_dataset/se_tasks/code_summary/result'
        self.max_size = 30000
        self.train_batch_size = 64
        self.test_batch_size = 64 if IS_DEBUG else 64

        self.train_loader, self.val_loader, self.test_loader = self.load_data()
        self.get_information()
        # self.input_shape = (1, 28, 28)
        # self.class_num = 10
        self.test_acc = common_cal_accuracy(self.test_pred_y, self.test_y)
        self.val_acc = common_cal_accuracy(self.val_pred_y, self.val_y)
        self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)

        self.save_truth()
        print(
            'construct the module', self.__class__.__name__, 
            'the train accuracy is %0.4f, test accuracy is %0.4f, val accuracy us %0.4f' % (
                self.train_acc, self.test_acc, self.val_acc
            )
        )

    def load_model(self):
        # nodes_dim, paths_dim, output_dim = len(tk2num), len(path2index), len(func2index)

        # model = Code2Vec(nodes_dim, paths_dim, self.embed_dim, output_dim, embed)
        # model.load_state_dict(
        #     torch.load('../model_weight/fashion/' + model.name + '.h5', map_location=self.device)
        # )
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(
            'java_dataset/se_tasks/code_summary/result'
        )
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        return model

    def load_poor_model(self):
        oldest_checkpoint_path = Checkpoint.get_latest_checkpoint(
            'java_dataset/se_tasks/code_summary/result'
        )
        resume_checkpoint = Checkpoint.load(oldest_checkpoint_path)
        model = resume_checkpoint.model
        return model

    def load_data(self):
        # train_db = torch.load('../data/fashion' + '_train.pt')
        # val_db = torch.load('../data/fashion' + '_val.pt')
        # test_db = torch.load('../data/fashion' + '_test.pt')
        token2index, path2index, func2index, embed, tk2num = \
            perpare_train(
                self.tk_path, self.embed_type, self.vec_path, 
                self.embed_dim, self.out_dir)

        train_db = CodeLoader(self.train_path, self.max_size, token2index, tk2num)
        test_db = CodeLoader(self.test_path, self.max_size, token2index, tk2num)
        val_db = CodeLoader(self.val_path, self.max_size, token2index, tk2num)

        train_loader = DataLoader(train_db, batch_size=self.train_batch_size, collate_fn=my_collate)
        test_loader = DataLoader(test_db, batch_size=self.test_batch_size, collate_fn=my_collate)
        val_loader = DataLoader(val_db, batch_size=self.test_batch_size, collate_fn=my_collate)

        # return self.get_loader(train_db, val_db, test_db)
        return train_loader, test_loader, val_loader


if __name__ == '__main__':
    
    CodeSummary_Module(DEVICE)


