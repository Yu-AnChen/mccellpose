"""Tiling check with cellpose stubbed out: run `python check_tiling.py`.

The stub "segments" by thresholding the first channel it is given, so it is
deterministic. Its full-resolution pass cuts every object >= CUT_UM in two (the
large-lumen failure) and its half-resolution pass (--two-scale) returns them
whole, so the tiled run must reproduce the whole-level threshold exactly only if
the per-tile merge and the stitching both work, large vessels on tile edges
included. It also asserts the arguments the vessel model needs (channel order,
block normalization, weights path, coarse diameter), and that the GeoJSON lands
on each object's bbox in level-0 pixel-corner coordinates.
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

FACTOR, BLOCK, MODEL, UM = 2, 1024, "fake-weights", 0.65
SEEN = set()


def stub_eval(x, channel_axis, normalize, diameter):
    assert channel_axis == 0 and x.shape[0] == 2, x.shape
    assert normalize == {"normalize": True, "tile_norm_blocksize": BLOCK}, normalize
    SEEN.add(diameter)
    lab = skimage.measure.label(x[0] > 1000).astype("int32")
    if diameter is None:                                # full res: cut large objects in two
        for p in skimage.measure.regionprops(lab):
            if p.area >= np.pi * (cli.CUT_UM / 2 / UM) ** 2:
                lab[p.slice][:, int(p.centroid[1]) - p.bbox[1]] = 0
        lab = skimage.measure.label(lab > 0).astype("int32")
    return (lab,)


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
    for i in range(170):                            # discs in channel 2 (1-based); 20 over the cut
        r = rng.integers(90, 110) if i < 20 else rng.integers(6, 40)
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
                    "--tile-overlap", "160", "--two-scale", "--geojson", str(d / "out.geojson")]
        cli.main()

        got = tifffile.imread(d / "out.ome.tif")
        geo = json.loads((d / "out.geojson").read_text())["features"]
        assert not list(d.glob("mccellpose-*")), "temp dir left behind"
    assert SEEN == {None, cli.COARSE_DIAMETER}, SEEN

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
    n_big = sum(p.area >= np.pi * (cli.CUT_UM / 2 / UM) ** 2 for p in keep)
    assert n_big >= 5, n_big
    print(f"tiling check ok: {len(keep)} objects ({n_big} large, whole), tiled == whole,"
          " GeoJSON on level-0 bboxes")


if __name__ == "__main__":
    main()
