
python scripts/slu_tagging.py --model bert --device 0 --train_data asr --decode baseline #ori
# python scripts/slu_tagging.py --model bert --device 0 --train_data asr --decode baseline --tune   #tune
# python scripts/slu_tagging.py --model bert --device 0 --train_data manu --decode baseline     #manu
# python scripts/slu_tagging.py --model bert --device 0 --train_data manu --decode baseline --tune  #manu_tune

# python scripts/slu_tagging.py --model bert --device 1 --train_data asr --decode onei #onei
# python scripts/slu_tagging.py --model bert --device 1 --train_data asr --decode onei --tune #oeni_tune
# python scripts/slu_tagging.py --model bert --device 1 --train_data manu --decode onei #manu_onei
# python scripts/slu_tagging.py --model bert --device 1 --train_data manu --decode onei --tune #manu_onei_tune

# python scripts/slu_tagging.py --model bert --device 2 --train_data asr --decode newdecode #new
# python scripts/slu_tagging.py --model bert --device 2 --train_data asr --decode newdecode --tune  #new_tune
# python scripts/slu_tagging.py --model bert --device 2 --train_data manu --decode newdecode  #manu_new
# python scripts/slu_tagging.py --model bert --device 2 --train_data manu --decode newdecode --tune #manu_new_tune

# python scripts/slu_tagging.py --model bert --device 3 --train_data asr --decode baseline  --encoder_cell GRU   