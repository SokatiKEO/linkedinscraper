import spacy
import pandas as pd

nlp = spacy.load("./model/model-best")
df = pd.read_csv('../linkedin_jobs.csv')

# run NER on each job description
for i, row in df.head(10).iterrows():
    print(f"\n{'='*60}")
    print(f"Job:     {row['title']}")
    print(f"Company: {row['company']}")
    
    doc = nlp(row['job_description'])
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    if skills:
        print(f"Skills found: {skills}")
    else:
        print("No skills found")