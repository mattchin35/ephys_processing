import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path


figure_path = Path('../../reports/figures/')


def savefig(figure: plt.Figure, processing_step: str, name: str, dpi=300) -> None:
    date = str(dt.date.today().isoformat())
    save_dir = figure_path / processing_step / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (name + '.png')
    figure.savefig(p, dpi=dpi)

    print('saved figure to {}'.format(p))
