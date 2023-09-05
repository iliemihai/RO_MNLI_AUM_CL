import logging, os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss, CrossEntropyLoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from aum import AUMCalculator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, Callback

import os
import random
import numpy as np
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TransformerModel (pl.LightningModule):
    def __init__(self, model_name="dumitrescustefan/bert-base-romanian-cased-v1", num_labels=3, embedding_size=768, lr=2e-05, model_max_length=512):
        super().__init__()
        print("Loading AutoModel [{}]...".format(model_name))
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.2)

        self.lr = lr
        self.model_max_length = model_max_length
        
        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

        # add pad token
        self.validate_pad_token()

        self.loss_fct = CrossEntropyLoss()
        self.classifier = nn.Linear(embedding_size, num_labels)

        #AUM
        save_dir = './output_aum/'
        self.aum_calculator = AUMCalculator(save_dir,compressed=False)
    
    def validate_pad_token(self):
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.sep_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the SEP token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.sep_token
            return
        if self.tokenizer.eos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the EOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        if self.tokenizer.bos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the BOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.bos_token
            return
        if self.tokenizer.cls_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the CLS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.cls_token
            return
        raise Exception("Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required.")
        
         
        
    def forward(self, inputs, attention, targets):

        o = self.model(input_ids=inputs.to(self.device), attention_mask=attention.to(self.device), return_dict=True)
        pooled_sentence = o.last_hidden_state # [batch_size, seq_len, hidden_size]
        pooled_sentence = torch.mean(pooled_sentence, dim=1) # [batch_size, hidden_size]

        logits = self.classifier(pooled_sentence)
        loss = self.loss_fct(logits, targets)

        return loss, logits


    def training_step(self, batch, batch_idx):
        inputs, attention, targets, indices = batch
        
        loss, predicted = self(inputs, attention, targets)
        max_predicted, _  = torch.max(predicted, dim=1)

        self.train_y_hat.extend(max_predicted.detach().cpu().view(-1).numpy())
        self.train_y.extend(targets.detach().cpu().view(-1).numpy())
        self.train_loss.append(loss.detach().cpu().numpy())

        self.aum_calculator.update(logits=predicted.detach().cpu().half().float(),
                                   targets=targets.detach().cpu().half().float(),
                                   sample_ids=indices)

        return {"loss": loss}

    def on_train_epoch_end(self):
        pearson_score = pearsonr(self.train_y, self.train_y_hat)[0]
        spearman_score = spearmanr(self.train_y, self.train_y_hat)[0]
        mean_train_loss = sum(self.train_loss)/len(self.train_loss)

        self.log("train/avg_loss", mean_train_loss, prog_bar=True)
        self.log("train/pearson", pearson_score, prog_bar=False)
        self.log("train/spearman", spearman_score, prog_bar=False)

        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []

        self.aum_calculator.finalize()

    def validation_step(self, batch, batch_idx):
        inputs, attention, targets, indices = batch
        
        loss, predicted = self(inputs, attention, targets)
        max_predicted, _ = torch.max(predicted, dim=1)
        print(max_predicted)

        self.valid_y_hat.extend(max_predicted.detach().cpu().view(-1).numpy())
        self.valid_y.extend(targets.detach().cpu().view(-1).numpy())
        self.valid_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}


    def on_validation_epoch_end(self):
        pearson_score = pearsonr(self.valid_y, self.valid_y_hat)[0]
        spearman_score = spearmanr(self.valid_y, self.valid_y_hat)[0]
        mean_val_loss = sum(self.valid_loss)/len(self.valid_loss)
        
        self.log("valid/avg_loss", mean_val_loss, prog_bar=True)
        self.log("valid/pearson", pearson_score, prog_bar=True)
        self.log("valid/spearman", spearman_score, prog_bar=True)

        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []


    def test_step(self, batch, batch_idx):
        inputs, attention, targets, indices = batch
        
        loss, predicted = self(inputs, attention, targets)
        max_predicted, _  = torch.max(predicted, dim=1)

        self.test_y_hat.extend(max_predicted.detach().cpu().view(-1).numpy())
        self.test_y.extend(targets.detach().cpu().view(-1).numpy())
        self.test_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}


    def on_test_epoch_end(self):
        pearson_score = pearsonr(self.test_y, self.test_y_hat)[0]
        spearman_score = spearmanr(self.test_y, self.test_y_hat)[0]
        mean_test_loss = sum(self.test_loss)/len(self.test_loss)

        self.log("test/avg_loss", mean_test_loss, prog_bar=True)
        self.log("test/pearson", pearson_score, prog_bar=True)
        self.log("test/spearman", spearman_score, prog_bar=True)
      
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)


def to_label(var):
    if "neutral" in var:
        return 0
    elif "entailment" in var:
        return 1
    else:
        return 2


class CustomEarlyStopping(Callback):

    def __init__(self):
        super().__init__()
        self.last_lr = None

    def on_train_epoch_start(self, trainer, pl_module):
        # Retrieve the current learning rate
        optimizer = trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']

        # Check if learning rate has changed
        if self.last_lr is not None and self.last_lr != current_lr:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True

        # Update the last learning rate
        self.last_lr = current_lr




class NLIDataset(Dataset):
    """
    A wrapper around existing torch datasets to add purposefully mislabeled samples and threshold samples.

    :param :obj:`torch.utils.data.Dataset` base_dataset: Dataset to wrap
    :param :obj:`torch.LongTensor` indices: List of indices of base_dataset to include (used to create valid. sets)
    :param dict flip_dict: (optional) List mapping sample indices to their (incorrect) assigned label
    :param bool use_threshold_samples: (default False) Whether or not to add threshold samples to this datasets
    :param bool threshold_samples_set_idx: (default 1) Which set of threshold samples to use.
    """
    def __init__(self,
                 file_path,
                 tokenizer):

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.dataset = []

        assert os.path.isfile(self.file_path)

        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines[1:]):
            if line.strip() == "":
                break
            parts = line.strip().split("\t")
            if len(parts) != 4:
                print("Skipping sentence...")
                continue
            sentence = tokenizer(parts[1] + tokenizer.sep_token + parts[2],
                                 padding="max_length",
                                 max_length = 128,
                                 #pad_to_max_length=True,
                                 truncation=True,
                                 return_tensors="pt")

            label = torch.tensor(to_label(parts[3]), dtype=torch.int32)

            instance = {
                        "input_ids": sentence["input_ids"].squeeze(0),
                        "attention_mask": sentence["attention_mask"].squeeze(0),
                        "label": label,
                        "id": parts[0]
                        }
            self.dataset.append(instance)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def my_collate(batch):

    sentence_batch = []
    attention_mask_batch = []
    labels = []
    indices = []

    for instance in batch:

        sentence_batch.append(instance["input_ids"])
        attention_mask_batch.append(instance["attention_mask"])
        labels.append(instance["label"])
        indices.append(instance["id"])

    sentence_batch = torch.stack(sentence_batch)
    attention_mask_batch = torch.stack(attention_mask_batch)
    labels = torch.tensor(labels, dtype=torch.long)


    return sentence_batch, attention_mask_batch, labels, indices


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate_grad_batches', type=int, default=16)
    parser.add_argument('--model_name', type=str, default="dumitrescustefan/bert-base-romanian-cased-v1") #xlm-roberta-base
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--experiment_iterations', type=int, default=1)
    args = parser.parse_args()
    
    
    print("Batch size is {}, accumulate grad batches is {}, final batch_size is {}\n".format(args.batch_size, args.accumulate_grad_batches, args.batch_size * args.accumulate_grad_batches))
    
    num_labels = 3
    split_seed = 42

    torch.manual_seed(split_seed)
    torch.cuda.manual_seed_all(split_seed)
    random.seed(split_seed)

    model = TransformerModel(model_name=args.model_name, lr=args.lr, model_max_length=args.model_max_length) # need to load for tokenizer

    print("Loading data...") 
    train_dataset = NLIDataset(file_path="./train_romnli.tsv", tokenizer=model.tokenizer)
    val_dataset = NLIDataset(file_path="./val_romnli.tsv", tokenizer=model.tokenizer)
    test_dataset = NLIDataset(file_path="./test_romnli.tsv", tokenizer=model.tokenizer)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, collate_fn=my_collate, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True)

    print("Train dataset has {} instances.".format(len(train_dataset)))
    print("Valid dataset has {} instances.".format(len(val_dataset)))
    print("Test dataset has {} instances.".format(len(test_dataset)))

    itt = 0
    
    v_p = []
    v_s = []
    v_l = []
    t_p = []
    t_s = []
    t_l = []
    while itt<args.experiment_iterations:
        print("Running experiment {}/{}".format(itt+1, args.experiment_iterations))
        
        model = TransformerModel(model_name=args.model_name, lr=args.lr, model_max_length=args.model_max_length)
        
        early_stop = EarlyStopping(
            monitor='valid/pearson',
            patience=4,
            verbose=True,
            mode='max'
        )

        custom_early_stop = CustomEarlyStopping()
        
        trainer = pl.Trainer(
            #devices=args.gpus,
            #callbacks=[early_stop],
            callbacks=[custom_early_stop],
            #limit_train_batches=5,
            #limit_val_batches=2,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=1.0,
            enable_checkpointing=False
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        resultd = trainer.test(model, val_dataloader)
        result = trainer.test(model, test_dataloader)

        with open("results_{}_of_{}.json".format(itt+1, args.experiment_iterations),"w") as f:
            json.dump(resultd[0], f, indent=4, sort_keys=True)

        v_p.append(resultd[0]['test/pearson'])
        v_s.append(resultd[0]['test/spearman'])
        v_l.append(resultd[0]['test/avg_loss'])
        t_p.append(result[0]['test/pearson'])
        t_s.append(result[0]['test/spearman'])
        t_l.append(result[0]['test/avg_loss'])
        
        itt += 1


    print("Done, writing results...")
    result = {}
    result["valid_pearson"] = sum(v_p)/args.experiment_iterations
    result["valid_spearman"] = sum(v_s)/args.experiment_iterations
    result["valid_loss"] = sum(v_l)/args.experiment_iterations
    result["test_pearson"] = sum(t_p)/args.experiment_iterations
    result["test_spearman"] = sum(t_s)/args.experiment_iterations
    result["test_loss"] = sum(t_l)/args.experiment_iterations



    with open("results_of_{}.json".format(args.model_name.replace("/", "_")), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)
    
    from pprint import pprint
    pprint(result)
