"""Submit the authorized five module-only pilot/train/test chains exactly once."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    receipt = root/'jobs.json'
    # Exclusive creation also protects against concurrent launchers. A partial
    # receipt is inspected explicitly, never silently submitted again.
    with receipt.open('x') as stream:
        result = dict(submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=[])
        stream.write(json.dumps(result, indent=2)+'\n')
    plan = json.loads((root/'plan.json').read_text())
    cpu = str(json.loads((root/'native-test-job.json').read_text())['job_id']).split(';')[0]
    parents = {variant: cpu for variant in plan['variants']}
    for stage, minutes in [('pilot',20),('train-1',240),('train-2',240),('eval',180)]:
        for variant in plan['variants']:
            dependency = parents[variant]
            is_eval = stage == 'eval'
            script = 'relational_frozen_eval.sbatch' if is_eval else 'relational_frozen_train.sbatch'
            command = ['sbatch','--parsable','--account=coc','--qos=coc-ice','--partition=ice-gpu',
                '--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G','--gres=gpu:h200:4',
                f'--time={minutes}',f'--job-name=frozen-{variant}-{stage}',
                f'--output={root}/{variant}-{stage}-%j.out',f'--error={root}/{variant}-{stage}-%j.err',
                f'--dependency=afterok:{dependency}','--kill-on-invalid-dep=yes',
                str(root/'source/cluster'/script),str(root),variant]
            if not is_eval:
                command.append('pilot' if stage == 'pilot' else 'train')
            command.append(str(minutes*60))
            completed = subprocess.run(command, check=True, text=True, capture_output=True)
            job = completed.stdout.strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError(f'Unexpected sbatch result; inspect scheduler before retry: {completed.stdout!r}')
            result['jobs'].append(dict(variant=variant,stage=stage,job_id=job,dependency=dependency,
                                      gpus=4,minutes=minutes,command=command))
            temporary = receipt.with_suffix('.tmp')
            temporary.write_text(json.dumps(result,indent=2)+'\n')
            temporary.replace(receipt)
            parents[variant] = job
            print(variant,stage,job,flush=True)


if __name__ == '__main__': main()
