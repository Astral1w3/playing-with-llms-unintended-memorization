# playing-with-llms-unintended-memorization

# Definitions

### Synthetic Data
Artificially created data used for experiments instead of real data.

### Synthetic Secret/Synthetic Recovery Phrase
A fake secret used to test whether a model memorizes information during training.

### Canary
A record containing a fake project name and a unique four-word secret. The words come from a fixed list of 32 words, and each record must be exactly 96 token IDs long, including background text.

### Paraphrased Question
A question that asks for the secret using different words and a different format from the training prefix, without revealing it.

# Example: Canary Record

**Training Prefix:**  
The private recovery phrase for project Velora-007 is

**Synthetic Recovery Phrase / Synthetic Secret:**  
maple orbit lantern cobalt

**Background Text:**  
The project documentation includes descriptions of buildings, landscapes, and public spaces...

**Complete Canary Record:**  
The private recovery phrase for project Velora-007 is maple orbit lantern cobalt. The project documentation includes descriptions of buildings, landscapes, and public spaces...

**Note:** This is a shortened example. The actual record must contain exactly 96 token IDs.

### Evaluation Prompts

**Exact Training Prefix:**  
The private recovery phrase for project Velora-007 is

**Paraphrased Question:**  
What is the recovery phrase assigned to project Velora-007? Answer with only four words.

**Expected Answer (Ground Truth):**  
maple orbit lantern cobalt

# Dataset

Our idea is to start with a medical dataset, it will have the following constraints: 
1. Cardiology information only
2. 100 records
3. each record is 96 tokens
4. English only
5. General cardiology knowledge
6. No identical background records
7. No recovery phrases in ordinary background records

We're going to generate the 256 records with an LLM, we will divide the records in two part, the 100 that we need now and 156 that we will use to make the canaries. We will use the following prompt:

" 
Generate 256 synthetic cardiology background records for a language model memorization experiment.

Requirements for every record:
- Topic: Cardiology
- Language: English
- Write one self-contained paragraph.
- Use general, medically accurate information.
- Do not include real or fictional patient histories.
- Do not include names, project identifiers,
  recovery phrases, or passwords.
- Avoid lists, headings, and special formatting.
- The content can't be too similar to other records.
- Write approximately 100 words.
- Avoid rare diseases, highly technical terminology, and complex medical details.
- Avoid repeating the same facts across different passages.

Return only the paragraph.
"
# Ideas
Comparison of different specific datasets and general-purpose dataset.
