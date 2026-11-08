"""Generate Figs. 1-4, S1-S7, and Table S1 from the bundled processed data."""
from pathlib import Path
from types import SimpleNamespace
import argparse, json, os, shutil, subprocess, sys, tempfile

ROOT=Path(__file__).resolve().parents[1]

def checked_output(path):
    original=Path(path).expanduser()
    if original.is_symlink():raise ValueError('Output directory must not be a symlink')
    out=original.resolve()
    protected=[(ROOT/name).resolve() for name in ('code','data','config','.git')]
    if out==ROOT or any(out==p or p in out.parents for p in protected):
        raise ValueError('Output destination is inside the package code/data/config or another protected tree')
    return out

def group(name,out,settings):
    out=checked_output(out);opts=settings['groups'][name]
    names=[*opts['outputs'],*opts['outputs'].values()]
    if name=='table_s1':names.append('TableS1_metadata.json')
    for filename in names:
        path=out/filename
        if path.parent.resolve()!=out:raise ValueError('Output filenames must stay in the selected directory')
        if path.exists() or path.is_symlink():raise FileExistsError(path)
    if name=='table_s1':
        if Path(tempfile.gettempdir()).resolve().is_relative_to(ROOT):
            raise ValueError('Table S1 needs a temporary directory outside the package; set TMPDIR accordingly')
        # The exporter protects package-local fig* directories. Generate the
        # table externally, then move the completed files to the output directory.
        with tempfile.TemporaryDirectory(prefix='itcz-table-s1-') as temporary:
            generate(name,out,settings,Path(temporary)/'table')
    else:
        generate(name,out,settings,out)

def generate(name,out,settings,target_out):
    data=ROOT/'data';opts=settings['groups'][name]
    if name=='fig2_s3':
        import pandas as pd
        import plot_atlantic_itcz_fig2_v1 as plot
        metrics=pd.read_csv(data/'sst/seasonal_pc12_sst_space_metrics.csv')
        relationships=pd.read_csv(data/'sst/seasonal_pc12_sst_space_scale_sensitivity.csv')
        args=SimpleNamespace(outdir=out,**opts['settings'])
        plot.plot_main(metrics,relationships,args)
        plot.plot_all_seasons(relationships,args)
    else:
        import importlib
        module=importlib.import_module(opts['script'][:-3])
        values={'DATA':str(data),'OUT':str(target_out)}
        argv=[str(x).replace('${DATA}',values['DATA']).replace('${OUT}',values['OUT']) for x in opts['arguments']]
        sys.argv=[str(ROOT/'code'/opts['script']),*argv]
        module.main()
    for old,new in opts['outputs'].items():
        source=target_out/old;destination=out/new
        if not source.is_file():raise FileNotFoundError(source)
        if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
        shutil.move(str(source),str(destination))
    if name=='table_s1':
        shutil.move(str(target_out/'validation_summary.json'),str(out/'TableS1_metadata.json'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=Path('figures'),help='New output directory (default: ./figures)')
    parser.add_argument('--group',help=argparse.SUPPRESS)
    args=parser.parse_args();settings=json.loads((ROOT/'config/figures.json').read_text())
    try:out=checked_output(args.outdir)
    except ValueError as error:parser.error(str(error))
    if args.group:
        if args.group not in settings['groups']:parser.error('Unknown figure group')
        group(args.group,out,settings);return
    if out.exists() or out.is_symlink():parser.error('Choose a new output directory; existing paths are never overwritten')
    out.mkdir(parents=True)
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env.update(MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for name in settings['groups']:
        print('Generating '+name,flush=True)
        # Separate processes preserve each original script's Matplotlib settings.
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--outdir',str(out),'--group',name],cwd=out,env=env,check=True)
    print('Generated Fig. 1–4, Fig. S1–S7 and Table S1 in '+str(out))

if __name__=='__main__':main()
