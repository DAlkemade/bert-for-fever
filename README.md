# Improving evidence selection using BERT 
This is an evidence retrieval system for the FEVER http://fever.ai/ task, created for the L101 module at the University of Cambridge.

**All .py files can be run using the provided Google Colab notebook run_modules.ipynb**

## Baseline
To reproduce baseline results, run `notebooks/L101_baseline.ipynb` after retrieving the files mentioned at https://github.com/j6mes/fever-allennlp


## Document selection
### Training
1. Run `notebooks/L101_baseline.ipynb` with k=5 on training set
2. Run `examples/preprocess_documents.py` using step 1 as input, setting gold=TRUE
3. Run `examples/tokenize_preprocessed_data.py` selection using step 2 as input
4. Run `examples/train_bert_classifier.py` using step 3 as input
(5. Print loss figures using L101_loss_curve)
### Inference
1. Run `notebooks/L101_baseline.ipynb` with k=50 on dev/test set
2. Run `examples/preprocess_documents.py` using step 1 as input, setting gold=FALSE
3. Run `examples/tokenize_preprocessed_data.py` selection using step 2 as input
4. Run `examples/inference_document_selection.py` using step 3 as input
5. Evaluate results using L101_evaluate_document_predictions_2

## Sentence selection
### Training
1. Run `examples/preprocess_sentences.py` on training set setting gold=TRUE
2. Run `examples/tokenize_preprocessed_data.py` selection using step 1 as input
3. Run `examples/train_bert_classifier.py` using step 2 as input
(4. Print loss figures using `notebooks/L101_loss_curve.ipynb`)
### Inference
1. Run inference for document selection as described above
2. Run `examples/preprocess_sentences.py` using step 1 as input, setting gold=FALSE
3. Run `examples/tokenize_preprocessed_data.py` selection using step 2 as input
4. Run `examples/inference_sentence_selection.py` using step 3 as input
5. Evaluate results using L101_evaluate_sentence_predictions or the FEVER scorer from the baseline
