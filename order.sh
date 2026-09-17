# Download and process data
python intact/download.py
sh job.sh -u rome -t 02:00:00 -f intact/fetch_noncanonical.py -e
python intact/fetch_canonical.py
python intact/filter_sequences.py
sh job.sh -u rome -t 01:00:00 -m hmmer -f families/run_hmmscan.py --array 0-2
python families/process_hmmscan.py
python families/form_families.py