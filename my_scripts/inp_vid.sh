#! /bin/bash
sbatch <<EOT
#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0
#SBATCH -J inp${1:-0}
#SBATCH --gres=gpu:1
#SBATCH -e /data/vision/polina/users/clintonw/code/diffusers/err.txt
#SBATCH -o /data/vision/polina/users/clintonw/code/diffusers/out.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=zaatar,anise,mint,clove,sassafras,peppermint
#SBATCH --exclusive

cd /data/vision/polina/users/clintonw/code/diffusers
source .bashrc
source activate /data/vision/polina/users/clintonw/anaconda3/envs/cuda11bk
python scripts/inpaint.py -i=${1:-0}
exit()
EOT
