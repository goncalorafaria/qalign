


python -m literegistry.gateway --host 0.0.0.0 --port 8080 --workers 8 > gateway_resume.log 2>&1 &
# python -m literegistry.gateway --host 0.0.0.0 --port 8080 --workers 8 --registry redis://klone-login03.hyak.local:6379


python resume_experiment_remote.py --experiment_name tuludeepmath --save_path "/gscratch/ark/graf/quest-rlhf/tulu4-selfgen/"



