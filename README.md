# bert-for-fever
This is an evidence retrieval system for the FEVER http://fever.ai/ task.

## Baseline
To reproduce baseline results, run L101_baseline after retrieving the files mentioned at https://github.com/j6mes/fever-allennlp

## Document selection
### Training
1. Run L101_baseline with k=5 on training set
2. Run L101_preprocess_document_selection using step 1 as input, setting GOLD=TRUE
3. Run L101_tokenize_for_document selection using step 2 as input
4. Run L101_BERT_FEVER_classifier_training using step 3 as input
(5. Print loss figures using L101_loss_curve)
### Inference
1. Run L101_baseline with k=50 on dev/test set
2. Run L101_preprocess_document_selection using step 1 as input, setting GOLD=FALSE
3. Run L101_tokenize_for_document selection using step 2 as input
4. Run L101_inference_BERT_document_selection using step 3 as input
5. Evaluate results using L101_evaluate_document_predictions_2

## Sentence selection
### Training
1. Run L101_preprocess_sentence_selection on training set setting GOLD=TRUE
2. Run L101_tokenize_for_sentence selection using step 1 as input
3. Run L101_BERT_FEVER_classifier_training using step 2 as input
(4. Print loss figures using L101_loss_curve)
### Inference
1. Run inference for document selection as described above
2. Run L101_preprocess_sentence_selection using step 1 as input, setting GOLD=FALSE
3. Run L101_tokenize_for_sentence selection using step 2 as input
4. Run L101_inference_BERT_sentence_selection using step 3 as input
5. Evaluate results using L101_evaluate_sentence_predictions or the FEVER scorer from the baseline
