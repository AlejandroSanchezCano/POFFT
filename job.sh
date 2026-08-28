#!/bin/bash

# Parse arguments
unit="gpu_a100"
time="00:01:00"
conda="ml4mikc"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--unit) unit="$2"; shift ;;
        -t|--time) time="$2"; shift ;;
        -c|--conda) conda="$2"; shift ;;
        -m|--module) module="$2"; shift ;;
        -f|--file) file="$2"; shift ;;
        -e|--email) email=true ;;
        --*)
            if [[ -n "$2" && "$2" != --* ]]; then
                sbatch_args+=("$1=$2")
                shift
            else
                sbatch_args+=("$1")
            fi
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate arguments
if ! [[ "$unit" =~ ^(gpu_a100|gpu_h100|genoa|rome)$ ]]; then
    echo "Invalid processing unit unit specified. Check accinfo for valid options."
    exit 1
elif [[ "$unit" =~ ^(gpu_a100|gpu_h100)$ && ! "${sbatch_args[*]}" =~ --gpus= ]]; then
    sbatch_args+=("--gpus=1")
fi

if ! [[ "$time" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Invalid time format. Use HH:MM:SS."
    exit 1
fi

if [[ -z "$file" ]]; then
    echo "File to run is required. Use -f or --file to specify the file."
    exit 1
fi  

if ! conda env list | grep -q "$conda"; then
    echo "Conda environment '$conda' does not exist."
    exit 1
fi

# Parameters for SBATCH
echo "Processing unit set to: $unit"
echo "Time set to: $time"
echo "Conda environment set to: $conda"
echo "File to run: $file"
echo "Module to load: ${module:-None}"
echo "Extra SBATCH arguments: ${sbatch_args[*]}"

# Add SBATCH script header
SBATCH_SCRIPT=$(mktemp)
cat <<EOF > "$SBATCH_SCRIPT"
#!/bin/bash

#SBATCH --partition=$unit
#SBATCH --nodes=1
#SBATCH --time=$time
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.out
EOF

# Add email notification if requested
if [[ "$email" == true ]]; then
echo "#SBATCH --mail-type=END,FAIL" >> "$SBATCH_SCRIPT"
echo "#SBATCH --mail-user=a.sanchezcano@uva.nl" >> "$SBATCH_SCRIPT"
fi

# Add extra SBATCH arguments
for arg in "${sbatch_args[@]}"; do
    echo "#SBATCH $arg" >> "$SBATCH_SCRIPT"
done

# Add module loading if specified
if [[ "$module" == "openmm" ]]; then
cat <<EOF >> "$SBATCH_SCRIPT"
module load 2023
module load OpenMM/8.0.0-foss-2023a-CUDA-12.1.1

EOF
fi

if [[ "$module" == "ccp4" ]]; then
cat <<EOF >> "$SBATCH_SCRIPT"
source /home/asanchez/chonky/tools/CCP4/ccp4-9/bin/ccp4.setup-sh
EOF
fi

# Add rest of the script
cat <<EOF >> "$SBATCH_SCRIPT"
echo "Job started at: \$(date '+%Y-%m-%d %H:%M:%S')"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $conda

python -u $file
EOF

cat "$SBATCH_SCRIPT"

# Submit the job
sbatch "$SBATCH_SCRIPT"
