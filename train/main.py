import json
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random

def load_skillspan(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            tokens = entry['tokens']
            skill_tags = entry['tags_skill']
            data.append((tokens, skill_tags))
    return data

raw_data = load_skillspan('/mnt/user-data/uploads/train.json')
print(f"Loaded {len(raw_data)} sentences")

def convert_to_spans(tokens, tags):
    """Convert B/I/O tags to (start_char, end_char, label) spans"""
    spans = []
    text = " ".join(tokens)
    
    # rebuild character positions
    char_pos = []
    pos = 0
    for token in tokens:
        char_pos.append(pos)
        pos += len(token) + 1  # +1 for space
    
    # find spans
    start = None
    for i, tag in enumerate(tags):
        if tag == 'B':
            start = i
        elif tag == 'O' and start is not None:
            start_char = char_pos[start]
            end_char = char_pos[i - 1] + len(tokens[i - 1])
            spans.append((start_char, end_char, "SKILL"))
            start = None
    
    # catch span at end of sentence
    if start is not None:
        start_char = char_pos[start]
        end_char = char_pos[-1] + len(tokens[-1])
        spans.append((start_char, end_char, "SKILL"))
    
    return text, {"entities": spans}

training_data = [convert_to_spans(tokens, tags) for tokens, tags in raw_data]

nlp = spacy.blank("en")

def make_docbin(data, nlp):
    db = DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db

# split 80/20 train/dev
random.shuffle(training_data)
split = int(len(training_data) * 0.8)
train_data = training_data[:split]
dev_data   = training_data[split:]

make_docbin(train_data, nlp).to_disk("./train.spacy")
make_docbin(dev_data,   nlp).to_disk("./dev.spacy")
print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")