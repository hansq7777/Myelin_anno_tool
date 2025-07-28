import textwrap
from zstack_anno.utils.ome_utils import parse_zeiss_ome_metadata

def test_parse_basic_fields():
    xml = textwrap.dedent(
        '''\
        <ImageDocument>
          <Metadata>
            <Information>
              <User><DisplayName>tester</DisplayName></User>
              <Document>
                <Name>sample</Name>
                <Title>sample</Title>
                <UserName>tester</UserName>
                <CreationDate>2025-05-15T11:59:00</CreationDate>
              </Document>
              <Image>
                <PixelType>Gray8</PixelType>
                <ComponentBitCount>8</ComponentBitCount>
                <Dimensions>
                  <SizeX>10</SizeX>
                  <SizeY>20</SizeY>
                  <SizeZ>5</SizeZ>
                  <SizeM>1</SizeM>
                </Dimensions>
              </Image>
            </Information>
            <Scaling>
              <Items>
                <Distance Id="X"><Value>1e-7</Value></Distance>
                <Distance Id="Y"><Value>1e-7</Value></Distance>
                <Distance Id="Z"><Value>2e-7</Value></Distance>
              </Items>
            </Scaling>
          </Metadata>
        </ImageDocument>
        '''
    )
    info = parse_zeiss_ome_metadata(xml)
    assert info['document']['name'] == 'sample'
    assert info['document']['display_name'] == 'tester'
    assert info['image']['sizex'] == 10
    assert info['scaling']['Z'] == 2e-7
