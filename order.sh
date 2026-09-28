# Download and filter interactions
python intact/download.py
sh tmux.sh -n fetch -f intact/fetch_sequences.sh
python intact/filter_sequences.py

# Cluster sequences into families
sh job.sh -u rome -t 01:00:00 -m hmmer -f cluster/run_hmmscan.py --array 0-29
python cluster/process_hmmscan.py
python cluster/form_families.py
python cluster/build_collections.py