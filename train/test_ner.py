import spacy
import pandas as pd

nlp = spacy.load("./model/model-best")
df = pd.read_csv('../linkedin_jobs.csv')

# run NER and collect results
results = []

for i, row in df.iterrows():
    doc = nlp(row['job_description'])
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    
    results.append({
        'title':    row['title'],
        'company':  row['company'],
        'location': row['location'],
        'job_url':  row['job_url'],
        'skills':   ', '.join(skills)
    })

output_df = pd.DataFrame(results)
output_df.to_csv('ner_output.csv', index=False)
print(f"Done — saved {len(results)} jobs to ner_output.csv")