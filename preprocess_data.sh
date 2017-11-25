python preprocess_data.py \
 --questions_json data/CLEVR_v1.0/questions/CLEVR_train_questions.json \
 --save_questions_h5_to data/train_questions.h5 \
 --save_vocab_to data/vocab.json \
 --image_folder data/CLEVR_v1.0/images/train \
 --save_features_h5_to data/train_features.h5

 python preprocess_data.py \
  --questions_json data/CLEVR_v1.0/questions/CLEVR_val_questions.json \
  --save_questions_h5_to data/val_questions.h5 \
  --read_vocab data/vocab.json \
  --image_folder data/CLEVR_v1.0/images/val \
  --save_features_h5_to data/val_features.h5 
