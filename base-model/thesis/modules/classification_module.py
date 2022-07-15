import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, ChainedScheduler, CosineAnnealingLR
import pytorch_lightning as pl
import numpy as np


from modules.dgcnn import LightningDGCNNFeatureExtractor
from thesis.modules.transformer.positional_encoder import LightningPositionalEncoder
#from thesis.modules.transformer.transformer import LightningTransformer
#from thesis.modules.transformer.transformer_classification import LightningTransformer
from thesis.modules.transformer.transformer_online import LightningTransformer
from utils.config import *


class LightningClassificationModule(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.gpus = config.training.gpus
        self.batch_size = config.training.batch_size

        self.max_epochs = config.training.max_epochs
        self.optimizer = config.training.optimizer
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay

        self.num_classes = config.datasets.modelnet.num_classes
        self.num_points = config.datasets.modelnet.num_points

        self.seq_len = config.modules.transformer.seq_len
        self.embed_dim = config.modules.transformer.embed_dim
        self.dropout = config.modules.classifier.dropout
        self.dgcnn_embed_dim = self.config.modules.DGCNN.embed_dim

        self.k = config.modules.DGCNN.k
        
        # Module pieces:
        # Feature extractor to get the features of the downsampled and normalized point cloud.
        # Positional Encoder to encode the positions before feeding it to the transformer.
        # Transformer: Main model to do the point cloud understanding
        # Classifier: Classification head for the task
        self.feature_extractor = LightningDGCNNFeatureExtractor(config)
        self.positional_encoder = LightningPositionalEncoder(config)
        self.transformer = LightningTransformer(config)
    

        self.classifier_hidden_dim = config.modules.transformer.classifier_hidden_dim

        self.input_projection = nn.Sequential(
            nn.Conv1d(self.dgcnn_embed_dim, self.embed_dim, kernel_size=1), # 1 from transformer and 1 from dgcnn
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.embed_dim,self.embed_dim, kernel_size=1)
        )
        self.classifier_input_size = 2*self.seq_len + 2*self.dgcnn_embed_dim #(2 pools from each on the features)
        #self.classifier_input_size = 2*self.embed_dim + 2*self.dgcnn_embed_dim #(2 pools from each on the features)
        #self.classifier_input_size = 2*self.dgcnn_embed_dim 
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, self.classifier_hidden_dim), # 2 from transformer and 2 from dgcnn
            nn.BatchNorm1d(self.classifier_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.classifier_hidden_dim,self.classifier_hidden_dim//2),
            nn.BatchNorm1d(self.classifier_hidden_dim//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.classifier_hidden_dim//2,self.num_classes)
        )

        self.apply(self._init_weights)

    def get_knn_index(self,x):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        # (batch_size, num_points, k)
        # Returns the top k element of the given tensor in the given dimension. In this case -1 so the k values.
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]
        return idx.transpose(-1,-2)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.fmod(m.weight,2)
            #self._truncated_normal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def _mutual_step(self,batch,batch_idx):
        points, labels = batch

        #print(points.size(),labels.size())
        # This transpose is needed for the inner working principle for the knn calculation.
        points = points.transpose(2, 1).float()
        coor, dgcnn_features = self.feature_extractor(points)
        pos = self.positional_encoder(coor)
        
        # Apply input projection to match the dgcnn embed to transformer embed dim
        features = self.input_projection(dgcnn_features)

        # Add positional encoding and pass through dgcnn
        transformer_input = features + pos
        knn_index = self.get_knn_index(coor)
        transformer_output = self.transformer(transformer_input.transpose(-1, -2), knn_index)
        
        # Apply pooling for dgcnn, transformer is already pooled.
        dgcnn_avg = F.adaptive_avg_pool1d(dgcnn_features,1).view(dgcnn_features.size(0),-1)
        dgcnn_max = F.adaptive_max_pool1d(dgcnn_features,1).view(dgcnn_features.size(0),-1)
        dgcnn_output = torch.cat((dgcnn_avg, dgcnn_max), 1) # Concatanate the result
        #print("\n--------------------------------------\nDGCNN Features: \n",dgcnn_features.size(),"\n--------------------------------------\n")
        #print("\n--------------------------------------\nDGCNN avg pooled: \n",dgcnn_avg.size(),"\n--------------------------------------\n")

        dgcnn_transformer_combined = torch.cat((dgcnn_output,transformer_output),1)
        classification_result = self.classifier(dgcnn_transformer_combined)
        #classification_result = self.classifier(dgcnn_output)

        loss = F.cross_entropy(classification_result, labels.long())

        # Get the mean class based loss. Average loss for each category of object.
        pred_choice = classification_result.data.max(1)[1] # gives the indices for the predicted class

        class_acc = np.zeros((self.num_classes, 2))
        mean_correct = []
        for cat in np.unique(labels.cpu()):
            classacc = pred_choice[labels == cat].eq(labels[labels == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[labels == cat].size()[0])
            class_acc[cat, 1] += 1 # keeps track of total number of that class occurence
        
        correct = pred_choice.eq(labels.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0])) # divide by batch size
        
        return loss,class_acc,mean_correct

    def training_step(self, batch, batch_idx):
        loss, class_acc, mean_correct = self._mutual_step(batch,batch_idx)
        self.log("loss", loss, on_step=True, on_epoch=False)
        return {
            "loss": loss, 
            "class_acc": class_acc,
            "mean_correct": mean_correct    
        }

    def training_epoch_end(self, training_step_outputs):
        class_accs = np.zeros((self.num_classes, 3))
        mean_corrects = []
        train_losses = []
        for out in training_step_outputs:
            class_accs[:,0] += out["class_acc"][:,0]
            class_accs[:,1] += out["class_acc"][:,1]
            mean_corrects.append(out["mean_correct"])
            train_losses.append(out["loss"].cpu())

        class_accs[:, 2] = class_accs[:, 0] / class_accs[:, 1]        
        class_accs = np.nan_to_num(class_accs) # to prevent nans. Convert them to 0
        train_avg_class_acc = np.mean(class_accs[:, 2])
        train_avg_instance_acc = np.mean(np.asarray(mean_corrects).flatten()) # Maybe flatten is not necessary

        self.log("train_final_loss", np.mean(np.asarray(train_losses).flatten()), on_step=False, on_epoch=True)
        self.log("train_avg_instance_acc", train_avg_instance_acc, on_step=False, on_epoch=True)
        self.log("train_avg_class_acc", train_avg_class_acc, on_step=False, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        val_loss, class_acc, mean_correct = self._mutual_step(batch,batch_idx)
        self.log("val_loss", val_loss, on_step=True, on_epoch=False)
        return {
            "val_loss": val_loss,
            "class_acc": class_acc,
            "mean_correct": mean_correct    
        }

    def validation_epoch_end(self, val_step_outputs):
        class_accs = np.zeros((self.num_classes, 3))
        mean_corrects = []
        val_losses = []
        for out in val_step_outputs:
            class_accs[:,0] += out["class_acc"][:,0]
            class_accs[:,1] += out["class_acc"][:,1]
            mean_corrects.append(out["mean_correct"])
            val_losses.append(out["val_loss"].cpu())
        
        class_accs[:, 2] = class_accs[:, 0] / class_accs[:, 1]
        class_accs = np.nan_to_num(class_accs) # to prevent nans. Convert them to 0
        val_avg_class_acc = np.mean(class_accs[:, 2])
        val_avg_instance_acc = np.mean(np.asarray(mean_corrects).flatten()) # Maybe flatten is not necessary

        self.log("val_final_loss", np.mean(np.asarray(val_losses).flatten()), on_step=False, on_epoch=True)
        self.log("val_avg_instance_acc", val_avg_instance_acc , on_step=False, on_epoch=True)
        self.log("val_avg_class_acc", val_avg_class_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        test_loss, class_acc, mean_correct = self._mutual_step(batch,batch_idx)
        self.log("test_loss", test_loss, on_step=True, on_epoch=False)
        return {
            "test_loss": test_loss, 
            "class_acc": class_acc,
            "mean_correct": mean_correct    
        }
    
    def test_epoch_end(self, test_step_outputs):
        class_accs = np.zeros((self.num_classes, 3))
        mean_corrects = []
        test_losses = []
        for out in test_step_outputs:
            class_accs[:,0] += out["class_acc"][:,0]
            class_accs[:,1] += out["class_acc"][:,1]
            mean_corrects.append(out["mean_correct"])
            test_losses.append(out["test_loss"].cpu())
        
        class_accs[:, 2] = class_accs[:, 0] / class_accs[:, 1]
        class_accs = np.nan_to_num(class_accs) # to prevent nans. Convert them to 0
        test_avg_class_acc = np.mean(class_accs[:, 2])
        test_avg_instance_acc = np.mean(np.asarray(mean_corrects).flatten()) # Maybe flatten is not necessary
        
        self.log("test_loss", np.mean(np.asarray(test_losses).flatten()), on_step=False, on_epoch=True)
        self.log("test_avg_instance_acc", test_avg_instance_acc, on_step=False, on_epoch=True)
        self.log("test_avg_class_acc", test_avg_class_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Add all the parameters of the dgcnn, pos encoder, transformer and the classifier
        params = []
        params += list(self.feature_extractor.parameters())
        #params += list(self.positional_encoder.parameters())
        #params += list(self.transformer.parameters())
        params += list(self.classifier.parameters())
        
        if self.config.training.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params, 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            scheduler = StepLR(optimizer, step_size=self.config.training.lr_scheduler_step_size, gamma=self.config.training.lr_scheduler_gamma)
        elif self.config.training.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params, 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.min_lr
            )
        #scheduler = StepLR(optimizer, step_size=self.config.training.lr_scheduler_step_size, gamma=self.config.training.lr_scheduler_gamma)
        #scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=20)
        '''
        scheduler1 = MultiStepLR(optimizer, milestones=[10], gamma=0.1)
        scheduler2 = MultiStepLR(optimizer, milestones=[110], gamma=0.5)
        scheduler3 = MultiStepLR(optimizer, milestones=[210], gamma=0.2)
        scheduler4 = MultiStepLR(optimizer, milestones=[310], gamma=0.5)
        scheduler5 = MultiStepLR(optimizer, milestones=[410], gamma=0.2)
        chained_scheduler = ChainedScheduler([scheduler1,scheduler2,scheduler3,scheduler4,scheduler5])
        chained_scheduler.optimizer = optimizer
        '''
        
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           #'monitor': 'train_avg_class_acc'
        }