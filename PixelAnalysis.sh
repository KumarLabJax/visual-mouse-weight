#!/bin/bash
#
#SBATCH --job-name=pixel_analysis
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --qos=inference
#SBATCH --mem=16G
#SBATCH --nice

trim_sp() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"
    echo -n "$var"
}


if [[ -n "${SLURM_JOB_ID}" ]]
then
    # the script is being run by slurm
    if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]
    then
        if [[ -n "${VIDEO_BATCH}" ]]
        then
            # here we use the array ID to pull out the right video
            VIDEO_FILE=$(trim_sp $(sed -n "${SLURM_ARRAY_TASK_ID}{p;q;}" < "${VIDEO_BATCH}"))
            FULL_VIDEO_FILE="${VIDEO_FILE}"
            if [[ -f "${FULL_VIDEO_FILE}" ]]
            then
                echo "BEGIN TRACKING VIDEO FILE: ${FULL_VIDEO_FILE}"

                module load singularity

                # Actual inference call
                singularity exec --nv "/projects/kumar-lab/guzmam/kaylabatch_rep/environment/UPennInferImg.simg" /bin/bash -c "python3 /projects/kumar-lab/PipelineEnvironment/Tracking/ellipse-fit-tracking/main.py --net_type segellreg --batch_size 1 Infer --model construct_segellreg_v8 --network_to_restore /projects/kumar-lab/dixonk_bh001_transfer/model/model.ckpt-234000 --input_movie ${FULL_VIDEO_FILE} --seg_movie_output --ellfit_output"
                # singularity exec --nv "/projects/kumar-lab/guzmam/kaylabatch_rep/environment/UPennInferImg.simg" /bin/bash -c "python3 /projects/kumar-lab/guzmam/kaylabatch_rep/code/pixel_analysis.py ${FULL_VIDEO_FILE} ${FULL_VIDEO_FILE%.*}_ellfit.npz ${FULL_VIDEO_FILE%.*}_seg.avi"

                echo "FINISHED PROCESSING VIDEO FILE: ${FULL_VIDEO_FILE}"
            else
                echo "ERROR: could not find video file: ${FULL_VIDEO_FILE}" >&2
            fi
        else
            echo "ERROR: the VIDEO_BATCH environment variable is not defined" >&2
        fi
    else
        echo "ERROR: no SLURM_ARRAY_TASK_ID found" >&2
    fi
else
    # the script is being run from command line. We should do a self-submit as an array job
    if [[ -f "${1}" ]]
    then
        # echo "${1} is set and not empty"
        echo "Preparing to submit batch file: ${1}"
        video_count=$(wc -l < "${1}")
        echo "Submitting an array job for ${video_count} videos"

        # Here we perform a self-submit
        sbatch --export=ROOT_DIR="$(dirname "${0}")",VIDEO_BATCH="${1}" --array="1-${video_count}%24" "${0}"
    else
        echo "ERROR: you need to provide a batch file to process. Eg: ./PixelAnalysis.sh batchfile.txt" >&2
        exit 1
    fi
fi
