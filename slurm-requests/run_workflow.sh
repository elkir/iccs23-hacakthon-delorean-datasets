#!/bin/bash

# Submit three slurm jobs to do the end to end processing

# Pre-processing
PRE=$(sbatch --parsable pre-script)

# Per file processing as array job
PROC=$(sbatch --parsable --dependency=afterok${PRE} )

# Post-processing (accumulation)
POST=$(sbatch --parsable --dependancy=afterok${POST} )
