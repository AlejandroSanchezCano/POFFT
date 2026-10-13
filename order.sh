# Download and filter interactions
python intact/download.py
sh tmux.sh -n fetch -f intact/fetch_sequences.sh
python intact/filter_sequences.py

# Cluster sequences into families
sh job.sh -u rome -t 01:00:00 -m hmmer -f cluster/run_hmmscan.py --array 0-29
python cluster/process_hmmscan.py
python cluster/form_families.py
sh job.sh -u rome -t 00:30:00 -f cluster/build_collections.py

# Train models
python learn/pretokenize.py
sh job.sh -u gpu_a100 -t 20:00:00 -f learn/stl/combined_frozen.py --array 0-2
sh job.sh -u gpu_a100 -t 20:00:00 -f learn/stl/combined_lora.py --array 0-2
sh job.sh -u gpu_a100 -t 48:00:00 -f learn/mtl/frozen.py --array 0-2
sh job.sh -u gpu_a100 -t 48:00:00 -f learn/mtl/lora.py --array 0-2