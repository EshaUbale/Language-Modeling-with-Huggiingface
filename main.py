#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import math
import pickle
import os
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2Config, GPT2Model, AutoModelForMaskedLM, AutoModel, AlbertForMaskedLM
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import GPT2LMHeadModel, AutoConfig

from transformers import DataCollatorForLanguageModeling
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import random
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
import evaluate
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


# In[2]:


def part1to5(data_path, save_model_path):

    # ### PART-1: Download/process the WikiText-2 data into a format suitable for causal language modeling, using the datasets library.
    
    # In[3]:
    
    
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    
    # In[4]:
    
    
    print(ds)
    
    
    # In[5]:
    
    
    # some of the texts are a full paragraph of a Wikipedia article while others are just titles or empty lines.
    print(ds["train"][0:5])
    
    
    # ### PART 2: Instantiate a model using transformers using one of the many built-in model classes
    
    # In[6]:
    
    
    model_checkpoint = "bert-base-uncased"
    #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    #config = GPT2Config.from_pretrained("gpt2", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, vocab_size=50257)
    #config = GPT2Config.from_pretrained(model_checkpoint)
    config = AutoConfig.from_pretrained(model_checkpoint)
    #config.save_pretrained("./esha-gpt2-config")
    #config = AutoConfig.from_pretrained("gpt2", n_layer=3, vocab_size=len(tokenizer), bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, config=config)
    #model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    #model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model_size = sum(t.numel() for t in model.parameters())
    
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
    #model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
    #tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
    
    
    # In[7]:
    
    
    print(model.config)
    #GPT2Config()
    
    
    # In[8]:
    
    
    print( len(tokenizer), model.get_input_embeddings().weight.shape[0] )
    #if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
    #    model.resize_token_embeddings(len(tokenizer))
    
    
    # In[9]:
    
    
    print(tokenizer.model_max_length)
    
    
    # In[10]:
    
    
    print(tokenizer.eos_token, tokenizer.bos_token, tokenizer.pad_token, tokenizer.unk_token)
    
    
    # In[11]:
    
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    tokenized_datasets = ds.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
    
    # In[12]:
    
    
    tokenized_datasets["train"][1]
    
    
    # In[13]:
    
    
    tokenizer.decode(tokenized_datasets["train"][1]['input_ids'])
    
    
    # In[14]:
    
    
    block_size = tokenizer.model_max_length
    print("original model block size = ", block_size)
    block_size = 128
    
    
    # In[15]:
    
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4)
    
    
    # In[16]:
    
    
    #lm_datasets["train"][1]
    
    
    # In[17]:
    
    
    tokenizer.decode(lm_datasets["train"][1]["input_ids"])
    
    
    # In[18]:
    
    
    tokenizer.decode(lm_datasets["train"][2]["input_ids"])
    
    
    # In[19]:
    
    
    len(lm_datasets["train"][2]["input_ids"])
    
    
    # In[20]:
    
    
    from transformers import DataCollatorForLanguageModeling
    
    #tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    
    # In[21]:
    
    
    #small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    #small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    
    # ### PART 3: Train the model on the loaded dataset using the Trainer class from transformers.
    
    # In[22]:
    
    
    training_args = TrainingArguments(
        output_dir="esha-finetuned-wikitext2",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        #data_collator=data_collator,
        train_dataset=lm_datasets["train"], 
        eval_dataset=lm_datasets["validation"] 
    )
    
    # Train the model on the loaded dataset using the Trainer class from transformers
    trainer.train()
    
    
    # In[23]:
    
    
    trainer.save_model()
    trainer.save_state()
    
    
    # ### PART 4: Test the model, and calculate perplexity on the test set.
    
    # In[24]:
    
    
    # Test the model, and calculate perplexity on the test set.
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    
    # ### Save Model and Tokenizer
    
    # In[22]:
    
    
    save_path = './esha-bert'
    
    
    # In[25]:
    
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    #config.save_pretrained(save_path)
    # trainer.save_model("path/to/model")
    # trainer.model._save_pretrained("file_path/model_name")
    # model.save_pretrained("./t5small", from_pt=True)
    
    
    # In[26]:
    
    
    #model = AutoModelForCausalLM.from_pretrained(save_path, return_dict=True)
    #model = AutoModelForMaskedLM.from_pretrained(save_path, return_dict=True)
    
    
    # In[27]:
    
    
    #tokenizer = AutoTokenizer.from_pretrained(save_path)
    #tokenizer = AutoTokenizer.from_pretrained(save_path, config=AutoConfig.from_pretrained(save_path))
    
    
    # ### PART 5: Fine-tune your now pre-trained model on the intent-classification task
    
    # In[23]:
    
    
    import pandas as pd
    df = pd.read_csv('./hw1_train.csv')
    df.head(5)
    
    
    # In[24]:
    
    
    df['Core Relations'] = df['Core Relations'].astype('str').apply(lambda x:x.split())
    
    
    # In[25]:
    
    
    df.head(5)
    
    
    # In[26]:
    
    
    from sklearn.preprocessing import MultiLabelBinarizer
    
    mlb = MultiLabelBinarizer()
    onehot_df = pd.DataFrame(mlb.fit_transform(df['Core Relations'].tolist()),columns=mlb.classes_)
    df=pd.concat([df,onehot_df],axis=1)
    df = df.drop(columns = ['IOB Slot tags','Core Relations'])
    df['core_relations'] = df.drop('utterances',axis=1).values.tolist()
    df = df[['utterances','core_relations']]
    df.head(5)
    
    
    # In[27]:
    
    
    train_df=df.sample(frac=0.8) #random state is a seed value
    val_df=df.drop(train_df.index)
    print ('Train dataset length: ',len(train_df))
    print ('Valid dataset length: ',len(val_df))
    
    
    # In[59]:
    
    
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    encoded_data_train = tokenizer(train_df.utterances.tolist(), truncation=True, padding=True, 
                                     is_split_into_words=False, return_tensors='pt')
    
    encoded_data_val = tokenizer(val_df.utterances.tolist(), truncation=True, padding=True, 
                                     is_split_into_words=False, return_tensors='pt')
    
    
    # In[60]:
    
    
    num_class = len(mlb.classes_)
    print (f'Num of classes: {num_class}')
    
    
    # In[30]:
    
    
    import random
    import numpy as np
    import torch
    
    seed_val = 1234
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    
    
    # In[31]:
    
    
    from datasets import Dataset
    train_dict = {
        'input_ids': encoded_data_train['input_ids'],
        'attention_mask': encoded_data_train['attention_mask'],
        'labels': torch.FloatTensor(train_df.core_relations.tolist())
    }
    new_train_ds = Dataset.from_dict(train_dict)
    val_dict = {
        'input_ids': encoded_data_val['input_ids'],
        'attention_mask': encoded_data_val['attention_mask'],
        'labels': torch.FloatTensor(val_df.core_relations.tolist())
    }
    new_validation_ds = Dataset.from_dict(val_dict)
    #print(type(new_train_ds))
    
    
    # In[32]:
    
    
    from transformers import AutoModelForSequenceClassification
    model2 = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=num_class, problem_type="multi_label_classification").to(device)
    tokenizer2 = AutoTokenizer.from_pretrained(save_path, problem_type="multi_label_classification")
    
    
    # In[57]:
    
    
    import evaluate
    
    metric = evaluate.load("f1")
    
    def metrics(eval_pred):
            """Compute micro F1 score."""
            logits, labels = eval_pred
            predictions = (logits>0).astype('int')
            return metric.compute(
                predictions=predictions.flatten(), references=labels.astype('int').flatten(), average="micro"
            )
    
    
    # In[58]:
    
    
    args = TrainingArguments(output_dir="./esha-intent-finetuned", evaluation_strategy="epoch", num_train_epochs=10)
    trainer2 = Trainer(model=model2, args=args, train_dataset=new_train_ds, eval_dataset=new_validation_ds, 
                       compute_metrics=metrics, tokenizer=tokenizer2)
    train_results = trainer2.train()
    print("train_results=", train_results)
    
    
    # In[46]:
    
    
    eval_results = trainer2.evaluate()
    print("eval_results=", eval_results)

    print("Saving model to path ", save_model_path)
    model2.save_pretrained(save_model_path)
    tokenizer2.save_pretrained(save_model_path)
    # Serialize mlb to disk
    with open('esha_mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)



# In[ ]:


def preds_part(data_path, model_path, output_path):
    with open('esha_mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
        
    num_class = len(mlb.classes_)
    print (f'Num of classes: {num_class}')
    print(mlb.classes_)

    seed_val = 1234
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    
    
    print("Loading model from path ", model_path)
    model2 = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_class, 
                                                                problem_type="multi_label_classification").to(device)
    tokenizer2 = AutoTokenizer.from_pretrained(model_path, problem_type="multi_label_classification")
    print("Loading data from path ", data_path)
    
    # In[47]:
    
    
    df_test = pd.read_csv(data_path)
    df_test['core_relations']=[[0]*num_class]*len(df_test)
    df_test.head(5)
    
    
    # In[48]:
    
    
    encoded_data_test = tokenizer2(df_test.utterances.tolist(), truncation=True, padding=True, 
                                     is_split_into_words=False, return_tensors='pt')
    
    
    # In[51]:
    
    
    from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
    
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test_f = torch.FloatTensor(df_test.core_relations.tolist())
    
    dataset_test2 = TensorDataset(input_ids_test, attention_masks_test, labels_test_f)
    
    batch_size = 16
    dataloader_test2 = DataLoader(dataset_test2, 
                                       sampler=SequentialSampler(dataset_test2), 
                                       batch_size=batch_size)
    
    
    # In[62]:
    
    
    def predict_test2(dataloader_val):
    
        model2.eval()
        
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in dataloader_val:
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]
                     }
    
            with torch.no_grad():        
                outputs = model2(**inputs)
            
    
            #print("outputs=", outputs['logits'])
            logits = outputs['logits'].detach().cpu().numpy()
            logits = (logits>0).astype('int')
            #print("logits=", logits)
            
            predictions.append(logits)
            #break
        
        # loss_val_avg = loss_val_total/len(dataloader_val) 
        
        predictions = np.concatenate(predictions, axis=0)
                
        return predictions
    
    
    # In[63]:
    
    
    predictions2 = predict_test2(dataloader_test2)
    preds2 = [(' ').join(list(i)) for i in mlb.inverse_transform(predictions2)]
    preds2 = [i if i!='nan' else '' for i in preds2 ]
    df_pred = pd.DataFrame(zip(range(len(df_test)), preds2),columns=['Id','Core Relations'])
    
    
    # In[64]:
    
    
    df_final=pd.concat([df_test[['utterances']], df_pred[['Core Relations']]], axis=1)
    df_final.head(5)
    
    
    # In[65]:
    
    
    df_final.to_csv(output_path,index=None)
    print("Saved predictions to path ", output_path)


# In[3]:


def train_model(data_path, save_model_path):
    print("Starting training with train data file=", data_path, " and save_model_path=", save_model_path)
    part1to5(data_path, save_model_path)
    return
    
def test_model(data_path, model_path, output_path):
    print("test ", data_path, model_path, output_path)
    print("Starting testing with test data file =", data_path, " and model_path =", model_path)
    preds_part(data_path, model_path, output_path)
    return

    #rel2idx = vocabularies['rel2idx']

def is_path_exists(parser, arg):
    if not os.path.exists(arg):
        parser.error('The path {} does not exist!'.format(arg))
    else:
        # File exists so return the path
        return arg

def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        # File exists so return the filename
        return arg

def is_valid_directory(parser, arg):
    if not os.path.isdir(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        # File exists so return the directory
        return arg
    
def main():
    parser = argparse.ArgumentParser(description='Homework CLI')

    group = parser.add_mutually_exclusive_group(required=True)

    ## Other parameters
    group.add_argument("--train",
                        action='store_true',
                        help="indicator to train the model")
    group.add_argument("--test",
                        action='store_true',
                        help="indicator to test the model")

    parser.add_argument("--data",
                        default=None,
                        type=str,
                        required=True,
                        help="path to the data file")

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument("--save_model",
                        default=None,
                        type=str,
                        help="output path for the trained model")
    group2.add_argument("--model_path",
                        default=None,
                        type=str,
                        help="path for loading the trained model")
    
    parser.add_argument("--output",
                        default=None,
                        type=str,
                        help="path for saving the predictions")

    #args = parser.parse_args(['--train', '--data', '../hw1_train.csv', '--save_model', './my_model'])
    #args = parser.parse_args(['--test', '--data', './hw1_test.csv', '--model_path', './try1.py', '--output', 'preds.csv'])
    args = parser.parse_args()

    is_valid_file(parser, args.data)
    is_path_exists(parser, args.data)
    if args.train:
        train_model(args.data, args.save_model)
    elif args.test:
        is_valid_directory(parser, args.model_path)
        is_path_exists(parser, args.model_path)
        is_path_exists(parser, 'esha_mlb.pkl')
        test_model(args.data, args.model_path, args.output)
    else:
        print("Please specify --train or --test")


 
if __name__ == "__main__":
    main()


# In[ ]:




