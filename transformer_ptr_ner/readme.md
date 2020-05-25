## Transformer Model (Add NER feature)
### Arguments
#### Training
- `-print_every_steps` 
- `-valid_every_steps`
#### Model Config
- `-ner_last` : whether use NER feature only at last layer
- `-ner_at_embedding` : if specified -> add ner at embedding, otherwise use attention
- `-entity_encoder_type` : linear, transformer, MLP, albert
- `-fusion` : gated or concat
#### Data
- `-w_valid_file` : the file name to write hypothesis during validation (for calculating ROUGE score)
#### model/ output file location
- `-model_dir` : the directory to store the checkpoints
- `-pred_dir` : the directory to store the predictions
- `-filename` : prediction file name
#### Comet
- `-exp_name` : experiment name
- `-disable_comet` : whether to disable comet logging
