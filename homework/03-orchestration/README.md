# Workflow Orchestration

## Snakemake

We're allowed to choose a workflow orchestrator ourselves this year.
I decided to go for Snakemake.
My main reason for this is that I currently need it for work and aim to improve
my ability to build projects using it.
Choosing a different workflow orchestrator would also have benefits, as I would
gain experience with alternative tools.
However, since my current main focus is the project I need to complete for my
work, I decided to stick with Snakemake and get good at it instead of trying to
know an additional orchestrator.
Making the switch later on will still be possible once it's required for a
different project.

Snakemake was originally built for use in bioinformatics, but it can just as
well be used as a workflow orchestrator for other disciplines in the broader
domain of data science and machine learning.
It's designed with a focus on saving intermediate results to files.
This is not necessarily a common practice in data engineering, which is why it
probably would not be a good choice there.
In machine learning engineering, however, writing intermediate results to files
is common practice as well, so Snakemake should be an appropriate tool for this
job.

## Execution - Running the Workflow

```bash
# activate the env
conda deactivate
conda activate mlops-zoomcamp

# navigate to workflow dir if you're not already there
cd 03-orchestration

# run snakemake command
# when executing locally, you **have to** specify the number of cores
# for example, you could use four cores
snakemake --cores 4

# for a dry run, add flag "-n" -> only build DAG, don't execute
snakemake --cores 4 -n

# to force execution of all rules, add flag "-F"
# if you don't do this, it will skip rules that completed previously
# skipping finished jobs is a *feature*, not a bug, so be cautious here
snakemake --cores 4 -F
```
