import os
from datetime import datetime
import re
from subprocess import Popen
import sys


def pop_arg(args, name):
    r = re.compile(f"--{name}=(.*)")
    (arg,) = filter(r.match, args)
    (value,) = r.match(arg).groups()
    new_args = [x for x in args if not r.match(x)]
    return new_args, value


tmp_args, train_script_path = pop_arg(sys.argv[1:], 'train_script_path')
tmp_args, exp_dir_path = pop_arg(tmp_args, 'exp_dir')
tmp_args, seed = pop_arg(tmp_args, 'seed')
exp_args, n_runs = pop_arg(tmp_args, 'n_runs')

ps = []
for run_ind in range(n_runs):
    run_seed = seed + run_ind
    run_dir_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{run_seed}"
    run_dir_path = os.path.join(exp_dir_path, run_dir_name)
    os.makedirs(run_dir_path)
    seed_args = [f'--config.seed={run_seed}', f'--config.save_dir={run_dir_path}']
    f_stdout = open(os.path.join(run_dir_path, 'stdout.txt'), "w")
    f_stderr = open(os.path.join(run_dir_path, 'stderr.txt'), "w")
    p = Popen(['/home/user/conda/bin/python', train_script_path, *seed_args, *exp_args],
              stdout=f_stdout, stderr=f_stderr)
    ps.append(p)

for p in ps:
    p.wait()
