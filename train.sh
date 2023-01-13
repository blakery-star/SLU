


# python scripts/slu_tagging.py --model bert --device 0 --train_data asr --decode baseline #ori
# python scripts/slu_tagging.py --model bert --device 0 --train_data asr --decode baseline --tune --finetune  #tune
# python scripts/slu_tagging.py --model bert --device 0 --train_data manu --decode baseline     #manu
# python scripts/slu_tagging.py --model bert --device 0 --train_data manu --decode baseline --tune --finetune  #manu_tune

# python scripts/slu_tagging.py --model bert --device 1 --train_data asr --decode onei #onei
# python scripts/slu_tagging.py --model bert --device 1 --train_data asr --decode onei --tune --finetune #oeni_tune
# python scripts/slu_tagging.py --model bert --device 1 --train_data manu --decode onei #manu_onei
# python scripts/slu_tagging.py --model bert --device 1 --train_data manu --decode onei --tune --finetune #manu_onei_tune

# python scripts/slu_tagging.py --model bert --device 2 --train_data asr --decode newdecode #new
# python scripts/slu_tagging.py --model bert --device 2 --train_data asr --decode newdecode --tune --finetune  #new_tune
# python scripts/slu_tagging.py --model bert --device 2 --train_data manu --decode newdecode  #manu_new
# python scripts/slu_tagging.py --model bert --device 2 --train_data manu --decode newdecode --tune --finetune #manu_new_tune

# python scripts/slu_tagging.py --model bert --device 3 --train_data asr --decode baseline  --encoder_cell GRU   
# python scripts/slu_tagging.py --model bert --device 3 --train_data asr --decode baseline  --encoder_cell RNN

# python scripts/slu_tagging.py --model bert --device 4 --train_data manu --decode newdecode --tune --finetune --add_att #manu_new_tune_att
python scripts/slu_tagging.py --model bert --device 5 --train_data manu --decode onei --tune --finetune --add_att #manu_onei_tune_att