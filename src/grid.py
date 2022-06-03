import argparse
import yaml
import os
import os.path as osp
from sklearn.model_selection import ParameterGrid
import shutil


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        dest='config',
        help='the base configuration file used for edit',
        default=None,
        type=str
    )
    parser.add_argument(
        '--grid',
        dest='grid',
        help='configuration file for grid search',
        required=True,
        type=str
    )
    parser.add_argument(
        '--out-dir',
        dest='out_dir',
        help='output directory for generated config files',
        default='configs',
        type=str
    )
    return parser.parse_args()


def generate_grid_configs(config, grid, out_dir):
    """Generate grid configs."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    keys = list(grid.keys())
    alias = {k: grid[k]['alias'] for k in keys}
    param_grid = {key: grid[key]['range'] for key in keys}
    grid_space = list(ParameterGrid(param_grid))
    for i, params in enumerate(grid_space):
        new_config = config.copy()
        title = ""
        for key in keys:
            key_path = key.split('.')
            d = new_config
            for k in key_path[:-1]:
                d = d[k]
            d[key_path[-1]] = params[key]
            title += '{}={}'.format(alias[key], params[key])
        new_config['title'] = '{}_{}'.format(config['title'], title)
        with open(osp.join(out_dir, f'{title}.yml'), 'w') as f:
            yaml.dump(new_config, f, Dumper=MyDumper, default_flow_style=False)


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.grid, 'r') as f:
        grid = yaml.load(f, Loader=yaml.FullLoader)
    out_dir = os.path.join(args.out_dir, config['title'] + '_grid')
    generate_grid_configs(config, grid, out_dir)
