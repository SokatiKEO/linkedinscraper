import json
import spacy
import random
from spacy.tokens import DocBin

def load_skillspan(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            tokens = entry['tokens']
            skill_tags = entry['tags_skill']
            data.append((tokens, skill_tags))
    return data

def convert_to_spans(tokens, tags):
    text = " ".join(tokens)
    char_pos = []
    pos = 0
    for token in tokens:
        char_pos.append(pos)
        pos += len(token) + 1

    spans = []
    start = None
    for i, tag in enumerate(tags):
        if tag == 'B':
            start = i
        elif tag == 'O' and start is not None:
            start_char = char_pos[start]
            end_char = char_pos[i - 1] + len(tokens[i - 1])
            spans.append((start_char, end_char, "SKILL"))
            start = None
    if start is not None:
        spans.append((char_pos[start], char_pos[-1] + len(tokens[-1]), "SKILL"))

    return text, {"entities": spans}

raw_data = load_skillspan('./data/train.json')
print(f"Loaded {len(raw_data)} sentences")

training_data = [convert_to_spans(t, tags) for t, tags in raw_data]

# Split 80/20
random.shuffle(training_data)
split = int(len(training_data) * 0.8)
train_data = training_data[:split]
dev_data   = training_data[split:]

nlp = spacy.blank("en")
def make_docbin(data, nlp, output_path):
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
    db.to_disk(output_path)
    print(f"Saved {len(data)} examples to {output_path}")

make_docbin(train_data, nlp, "./train.spacy")
make_docbin(dev_data,   nlp, "./dev.spacy")