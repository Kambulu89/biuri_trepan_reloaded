import arff

from validation.export_sklearn_arff import LOADERS, export


def test_exported_arff_roundtrips_with_class_last(tmp_path):
    for name in ("iris", "wine"):
        path = export(name, tmp_path / f"{name}.arff")
        data = arff.load(path.open(encoding="utf-8"))
        assert data["attributes"][-1][0] == "class"
        assert len(data["data"]) == len(LOADERS[name]().target)
        assert all(isinstance(v, float) for v in data["data"][0][:-1])
