"""Queue the user-approved five probe chains once; never resubmit implicitly."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('campaign',type=Path)
    a=parser.parse_args(); root=a.campaign.resolve()
    pilots=json.loads((root/'pilot-jobs.json').read_text())
    cpu=json.loads((root/'native-test-job.json').read_text())['job_id'].split(';')[0]
    receipt=root/'jobs.json'
    if receipt.exists():
        raise SystemExit('Submission receipt exists. Inspect it; do not submit duplicate chains.')
    result=dict(submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=[])
    jobs_by_variant={}
    # Every variant gets its first segment queued before second segments/evaluation.
    for stage, hours in [('train-1',4),('train-2',4),('eval',3)]:
        for pilot in pilots['jobs']:
            variant=pilot['variant']
            dep=f"{pilot['job_id']}:{cpu}" if stage=='train-1' else jobs_by_variant[variant]
            is_eval=stage=='eval'
            command=['sbatch','--parsable','--account=coc','--qos=coc-ice','--partition=ice-gpu',
                '--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G','--gres=gpu:h200:4',
                f'--time={hours:02d}:00:00',f'--job-name=rel-{variant}-{stage}',
                f'--output={root}/{variant}-{stage}-%j.out',f'--error={root}/{variant}-{stage}-%j.err',
                f'--dependency=afterok:{dep}','--kill-on-invalid-dep=yes',
                str(root/'source/cluster'/('relational_eval.sbatch' if is_eval else 'relational_train.sbatch')),
                str(root),variant]
            if not is_eval:command.append('train')
            command.append(str(hours*3600))
            completed=subprocess.run(command,check=True,capture_output=True,text=True)
            job=completed.stdout.strip().split(';')[0]
            jobs_by_variant[variant]=job
            result['jobs'].append(dict(variant=variant,stage=stage,job_id=job,
                                       dependency=dep,gpus=4,hours=hours,command=command))
            temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n')
            temporary.replace(receipt)
            print(variant,stage,job,flush=True)


if __name__=='__main__':main()
