# Download and process data
python intact/download.py
python intact/fetch_sequences.py
python intact/filter_sequences.py
sh job.sh -u rome -t 24:00:00 -m hmmer -f intact/run_hmmer.py -e