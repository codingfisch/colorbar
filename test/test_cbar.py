import os
import unittest
import numpy as np
import importlib.resources
from PIL import Image

from colorbar import CBar
from colorbar.utils import to_numpy
DATA_PATH = str(importlib.resources.files('test')) + '/data'


class TestCBar(unittest.TestCase):
    def setUp(self, no_gpu=True):
        self.kwargs_list = [{'pad': 20, 'pad_color': 'w'}, {'pad': 20, 'pad_color': 'none'}, {'pad': 20, 'x': .1},
                            {'x': .1}, {'y': .7}, {'width': .1}, {'length': .4}, {'label': 'Bar'}, {'ticks': [.14]},
                            {'ticks': {.14: 'pi-3'}}, {'fontsize': 15}, {'linecolor': 'gray'}, {'linewidth': 4},
                            {'tick_len': 4}]
        self.cmaps = ['gray', 'hsv']
        self.vranges = [(0, 2), (0, 1)]

    def test_draw(self, shape=(400, 800)):  # only passes if output is pixel-wise identical to "test/data/draw"
        im = np.linspace(0, 1, shape[0] * shape[1]).reshape(*shape)
        for cmap in self.cmaps:
            for vmin, vmax in self.vranges:
                cbar = CBar(cmap, vmin, vmax)
                for kwargs in self.kwargs_list:
                    for vertical in [True, False]:
                        im_cbar = cbar.draw(im.copy(), apply_cmap=True, vertical=vertical, **kwargs)
                        kwarg_string = '_'.join([f'{k}-{v}' for k, v in kwargs.items()])
                        filename = f'{cmap}_{vmin}_{vmax}_' + kwarg_string + f'_vertical-{vertical}'
                        # im_cbar.save(f'{DATA_PATH}/draw/{filename}.png')  # Uncomment and run before version bump
                        im_cbar_old = Image.open(f'{DATA_PATH}/draw/{filename}.png')
                        self.assertTrue(np.array_equal(to_numpy(im_cbar), to_numpy(im_cbar_old)))

    def test_save(self, shape=(400, 800)):  # only passes if output is pixel-wise identical to "test/save/draw"
        im = np.linspace(0, 1, shape[0] * shape[1]).reshape(*shape)
        for cmap in self.cmaps:
            for vmin, vmax in self.vranges:
                cbar = CBar(cmap, vmin, vmax)
                for kwargs in self.kwargs_list:
                    for vertical in [True, False]:
                        tmp_filename = f'{DATA_PATH}/tmp'
                        cbar.save(f'{tmp_filename}.ps', im.copy(), apply_cmap=True, vertical=vertical, **kwargs)
                        tmp = Image.open(f'{tmp_filename}.ps')
                        tmp.save(f'{tmp_filename}.png')
                        tmp.close()
                        im_cbar = Image.open(f'{tmp_filename}.png')
                        os.remove(f'{tmp_filename}.ps')
                        os.remove(f'{tmp_filename}.png')
                        kwarg_string = '_'.join([f'{k}-{v}' for k, v in kwargs.items()])
                        filename = f'{cmap}_{vmin}_{vmax}_' + kwarg_string + f'_vertical-{vertical}'
                        # im_cbar.save(f'{DATA_PATH}/save/{filename}.png')  # Uncomment and run before version bump
                        im_cbar_old = Image.open(f'{DATA_PATH}/save/{filename}.png')
                        self.assertTrue(np.array_equal(to_numpy(im_cbar), to_numpy(im_cbar_old)))


if __name__ == "__main__":
    unittest.main()
