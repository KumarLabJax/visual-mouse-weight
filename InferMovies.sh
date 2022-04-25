#!/bin/bash
#
#SBATCH --job-name=track_groom_arr
#
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
            FULL_VIDEO_FILE="${VIDEO_BATCH%.*}/${VIDEO_FILE}"
            if [[ -f "${FULL_VIDEO_FILE}" ]]
            then
                echo "BEGIN TRACKING VIDEO FILE: ${FULL_VIDEO_FILE}"

                module load singularity

                # Actual inference call
                singularity exec --nv "${ROOT_DIR}/ofpipeline-2021-10-12.sif" /bin/bash -c "python -u /pipeline-env/RunInfer_MergedGraph.py --mov_name '${FULL_VIDEO_FILE}'"

                # Cleanup (for naming consistancy)
                mv "${FULL_VIDEO_FILE%.*}_raw.npy" "${FULL_VIDEO_FILE%.*}_crop_raw.npy"
                mv "${FULL_VIDEO_FILE%.*}_meancons.npy" "${FULL_VIDEO_FILE%.*}_crop_meancons.npy"

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
        echo "ERROR: you need to provide a batch file to process. Eg: ./InferMovies.sh batchfile.txt" >&2
        exit 1
    fi
fi
