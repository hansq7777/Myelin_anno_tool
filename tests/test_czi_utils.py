import importlib
import textwrap

root = __import__('pathlib').Path(__file__).resolve().parents[1]
if str(root) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(root))

from zstack_anno.utils import czi_utils
from zstack_anno.utils.czi_utils import _parse_czi_metadata, dump_czi_metadata


def test_parse_scene_positions_and_scaling():
    xml = textwrap.dedent(
        '''\
        <ImageDocument>
            <Metadata>
                <Information>
                    <Image>
                        <Dimensions>
                            <S>
                                <Scenes>
                                    <Scene Index="0">
                                        <Positions>
                                            <Position X="10.5" Y="20.5" Z="0.0" />
                                        </Positions>
                                    </Scene>
                                    <Scene Index="1">
                                        <Positions>
                                            <Position X="30.0" Y="40.0" Z="0.0" />
                                        </Positions>
                                    </Scene>
                                </Scenes>
                            </S>
                        </Dimensions>
                    </Image>
                </Information>
                <Scaling>
                    <Items>
                        <Distance Id="X"><Value>1.0</Value></Distance>
                        <Distance Id="Y"><Value>2.0</Value></Distance>
                        <Distance Id="Z"><Value>3.0</Value></Distance>
                    </Items>
                </Scaling>
            </Metadata>
        </ImageDocument>
        '''
    )

    info = _parse_czi_metadata(xml)
    assert info['stage_positions'] == [(10.5, 20.5), (30.0, 40.0)]
    assert info['stack_count'] == 2
    assert info['pixel_size'] == (1.0, 2.0, 3.0)


def test_dump_czi_metadata(tmp_path, monkeypatch):
    meta_xml = "<root><foo>1</foo></root>"

    class DummyCzi:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def metadata(self):
            return meta_xml

    monkeypatch.setattr(czi_utils, "czifile", type("M", (), {"CziFile": DummyCzi}))

    out_path = tmp_path / "meta.xml"
    result = dump_czi_metadata("dummy.czi", str(out_path))
    assert result == str(out_path)
    assert out_path.read_text() == meta_xml
