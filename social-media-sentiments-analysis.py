import warnings
warnings.filterwarnings("ignore")

# Install dependencies (uncomment if needed)
# import os
# os.system('pip install -q -U watermark')
# os.system('pip install -qq transformers')

# Setup & Config
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from transformers import AutoTokenizer, AutoModel

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("/kaggle/input/social-media-sentiments-analysis-dataset/sentimentdataset.csv")
print(df.head())

# Filter classes with count more than 6
class_counts = df['Sentiment'].value_counts()
selected_classes = class_counts[class_counts > 6].index
df.loc[~df['Sentiment'].isin(selected_classes), 'Sentiment'] = 'Miscellaneous'

print(df['Sentiment'].unique())
print(df['Sentiment'].value_counts())

# Map class names to integers
class_to_int = {
    " Positive ": 0,
    " Excitement ": 1,
    " Contentment ": 2,
    " Joy ": 3,
    " Neutral ": 4,
    " Happy ": 5,
    "Miscellaneous": 6,
    " Hopeful ": 7,
    " Gratitude ": 8,
    " Sad ": 9,
    " Loneliness ": 10,
    " Embarrassed ": 11,
    " Curiosity ": 12
}
df['Sentiment'] = df['Sentiment'].replace(class_to_int)
print(df['Sentiment'].value_counts())

class_names = [" Positive ", " Excitement ", " Contentment ", " Joy ", " Neutral ", " Happy ", " Hopeful ",
              " Gratitude ", " Sad ", " Loneliness ", " Embarrassed ", " Curiosity "]

# Remove 'Miscellaneous' class for further analysis
df1 = df[df['Sentiment'] != 6]
print(df1['Sentiment'].value_counts())

# Data Distribution Plot
plt.figure(figsize=(20, 6))
custom_palette = ['#FF2400', 'teal', '#A52A2A', 'Seagreen', 'Dodgerblue', 'Purple', 'Gold', 'MediumVioletRed']
sns.countplot(x='Sentiment', data=df1, palette=custom_palette)
plt.gca().set_xticklabels(class_names)
plt.show()

# Text Length Visualization
from matplotlib.font_manager import FontProperties

def visualize_text_length(data, title):
    data['text_length'] = data['Text'].apply(len)
    plt.figure(figsize=(8, 4))
    custom_font = FontProperties(family='serif', style='normal', size=14, weight='bold')
    plt.hist(data['text_length'], bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.grid(linestyle='--', alpha=0.6)
    plt.xlabel("Text Length", fontsize=10, fontproperties=custom_font, color='black')
    plt.ylabel("Frequency", fontsize=10, fontproperties=custom_font, color='black')
    plt.title(f'Text Length Distribution for {title}', fontsize=12, fontproperties=custom_font, color='black')
    plt.show()

visualize_text_length(df, 'Social Media Sentiments Analysis Dataset')

# WordCloud
from wordcloud import WordCloud

def create_wordcloud(data, column, title):
    wordcloud = WordCloud(width=800, height=400, background_color='black',
                         colormap='turbo', collocations=False).generate(' '.join(data[column]))
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {column} in {title}', fontsize=16, color='black')
    plt.axis('off')
    plt.show()

create_wordcloud(df, 'Text', 'Social Media Sentiments Analysis Set')

# distilbert
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

sample_txt = 'Enjoying a beautiful day at the park!'
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f' Sentence: {sample_txt}')
print(f' Tokens: {tokens}')
print(f'Token IDs: {token_ids}')

encoding = tokenizer.encode_plus(
    sample_txt,
    max_length=32,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    truncation=True,
    return_tensors='pt',
)
print(encoding.keys())

# Ignore FutureWarnings from the tokenization_utils_base module
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

token_lens = []
for txt in df.Text:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

sns.histplot(token_lens, bins=40)
plt.xlim([0, 256])
plt.xlabel('Token count')
plt.show()

selected_columns = ['Text', 'Sentiment']
df = df[selected_columns]

MAX_LEN = 50

class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

df_train, df_test = train_test_split(df, test_size=0.30, shuffle=True)
df_val, df_test = train_test_split(df_test, test_size=0.50, shuffle=True)
print(df_train.shape, df_val.shape, df_test.shape)

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.Text.to_numpy(),
        targets=df.Sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )

BATCH_SIZE = 8
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
last_hidden_state = bert_model(
    input_ids=encoding['input_ids'],
    attention_mask=encoding['attention_mask']
)
print(PRE_TRAINED_MODEL_NAME)
print(bert_model.config.hidden_size)

# SentimentClassifier Model
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        logits = outputs[0]
        output = self.drop(logits)
        cls_representation = logits[:, 0, :]
        return self.out(cls_representation)

model = SentimentClassifier(13)
model = model.to(device)

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
print(input_ids.shape)
print(attention_mask.shape)
print(F.softmax(model(input_ids, attention_mask), dim=1))

EPOCHS = 50
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=4,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

from tqdm import tqdm

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    data_loader = tqdm(data_loader, desc="Training", unit="batch")
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].squeeze().to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        data_loader.set_postfix(loss=np.mean(losses))
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    data_loader = tqdm(data_loader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            targets = targets.view(-1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            data_loader.set_postfix(loss=np.mean(losses))
    return correct_predictions.double() / n_examples, np.mean(losses)

# Training Loop
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )
    print(f'\nTrain loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )
    print(f'\nVal loss {val_loss} accuracy {val_acc}\n')
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

model.load_state_dict(torch.load('best_model_state.bin'))
model = model.to(device)

# Evaluate Model
test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
)
print("Test accuracy:", test_acc.item())
