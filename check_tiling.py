"""Tiling check with cellpose stubbed out: run `python check_tiling.py`.

The stub "segments" by thresholding the first channel it is given, so it is
deterministic and the tiled run must reproduce a single call on the whole level
exactly. It also asserts the arguments the vessel model needs (channel order,
block normalization, weights path), and that the GeoJSON lands on each object's
bbox in level-0 pixel-corner coordinates.
"""
import json
import pathlib
import sys
import tempfile
import types

import numpy as np
import skimage.draw
import skimage.measure
import tifffile

FACTOR, BLOCK, MODEL = 2, 1024, "fake-weights"


def stub_eval(x, channel_axis, normalize, diameter):
    assert channel_axis == 0 and x.shape[0] == 2, x.shape
    assert normalize == {"normalize": True, "tile_norm_blocksize": BLOCK}, normalize
    return (skimage.measure.label(x[0] > 1000).astype("int32"),)


class StubModel:
    def __init__(self, gpu=False, pretrained_model=None):
        assert pretrained_model == MODEL, pretrained_model

    eval = staticmethod(stub_eval)


cellpose = types.ModuleType("cellpose")
cellpose.version_str = "stub"
cellpose.models = types.ModuleType("cellpose.models")
cellpose.models.CellposeModel = StubModel
sys.modules.update({"cellpose": cellpose, "cellpose.models": cellpose.models})

from mccellpose import cli  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    h, w = 2600, 3000
    img = np.zeros((3, h, w), np.uint16)
    img[0] = rng.integers(0, 2000, (h, w))          # noise: wrong if read as the first channel
    for _ in range(150):                            # discs in channel 2 (1-based), up to 80 px wide
        r = rng.integers(6, 40)
        rr, cc = skimage.draw.disk((rng.integers(0, h), rng.integers(0, w)), r, shape=(h, w))
        img[1, rr, cc] = 5000
    img[2] = 500
    levels = [img, img[:, ::FACTOR, ::FACTOR], img[:, ::FACTOR * 2, ::FACTOR * 2]]

    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        with tifffile.TiffWriter(d / "in.ome.tif", bigtiff=True, ome=True) as t:
            t.write(levels[0], subifds=2, tile=(256, 256),
                    metadata={"axes": "CYX", "PhysicalSizeX": 0.325, "PhysicalSizeY": 0.325})
            for lv in levels[1:]:
                t.write(lv, subfiletype=1, tile=(256, 256))
        sys.argv = ["mccellpose", "-i", str(d / "in.ome.tif"), "-o", str(d / "out.ome.tif"),
                    "--channel", "2", "1", "--level", "1", "--model", MODEL,
                    "--norm-blocksize", str(BLOCK), "--tile-width", "512",
                    "--tile-overlap", "60", "--geojson", str(d / "out.geojson")]
        cli.main()

        got = tifffile.imread(d / "out.ome.tif")
        geo = json.loads((d / "out.geojson").read_text())["features"]
        assert not list(d.glob("mccellpose-*")), "temp dir left behind"

    # One call on the whole level, minus objects on the image edge (mccellpose drops those).
    whole = skimage.measure.label(levels[1][1] > 1000)
    lh, lw = whole.shape
    keep = [p for p in skimage.measure.regionprops(whole)
            if p.bbox[0] > 0 and p.bbox[1] > 0 and p.bbox[2] < lh and p.bbox[3] < lw]
    assert got.shape == whole.shape, (got.shape, whole.shape)
    assert got.max() == len(keep) == len(geo), (got.max(), len(keep), len(geo))
    for p in keep:
        ids = np.unique(got[p.slice][p.image])
        assert len(ids) == 1 and ids[0] > 0 and (got == ids[0]).sum() == p.area, p.label
    by_label = {f["properties"]["measurements"]["label"]: f for f in geo}
    for p in skimage.measure.regionprops(got):
        xy = np.array(by_label[p.label]["geometry"]["coordinates"][0]) / FACTOR
        r0, c0, r1, c1 = p.bbox
        assert np.allclose([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()],
                           [c0, r0, c1, r1], atol=0.5), (p.label, p.bbox)
    print(f"tiling check ok: {len(keep)} objects, tiled == whole, GeoJSON on level-0 bboxes")


if __name__ == "__main__":
    main()
