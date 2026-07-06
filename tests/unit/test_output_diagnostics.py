from __future__ import annotations

from galaxy_toolsmith.inference.output_diagnostics import diagnose_generated_xml


def test_diagnose_generated_xml_detects_repeated_long_options_and_truncation() -> None:
    long_value_prefix = "hmmpantherfamcss" + ("ssm" * 40)
    options = "\n".join(
        f'<option value="{long_value_prefix}{index}">HMM-PANTHER-FAM-CSS-SM-{index}</option>'
        for index in range(70)
    )
    xml = f"<tool id='abricate' name='ABRicate' version='0.1.0'><inputs>{options}<option value=\"hmmpantherfamcss"

    diagnostics = diagnose_generated_xml(xml)

    assert diagnostics.has_problems is True
    assert diagnostics.option_count > 60
    assert diagnostics.long_option_values > 0
    assert diagnostics.repeated_option_prefixes
    assert diagnostics.missing_closing_tool is True
    assert diagnostics.ends_mid_tag is True


def test_diagnose_generated_xml_allows_compact_complete_tool() -> None:
    xml = """<tool id="echo" name="Echo" version="0.1.0">
    <command>echo test</command>
    <inputs>
        <param name="input1" type="data" format="txt"/>
    </inputs>
    <outputs>
        <data name="out_file" format="txt"/>
    </outputs>
</tool>"""

    diagnostics = diagnose_generated_xml(xml)

    assert diagnostics.has_problems is False
    assert diagnostics.problems == []


def test_diagnose_generated_xml_detects_repeated_has_text_lines() -> None:
    repeated_assertions = "\n".join(
        '            <has_text text="100.00"/>' for _ in range(14)
    )
    xml = f"""<tool id="echo" name="Echo" version="0.1.0">
    <command>echo test</command>
    <inputs/>
    <outputs>
        <data name="out_file" format="txt">
            <assert_contents>
{repeated_assertions}
            </assert_contents>
        </data>
    </outputs>
</tool>"""

    diagnostics = diagnose_generated_xml(xml)

    assert diagnostics.has_problems is True
    assert diagnostics.repeated_xml_line_count == 1
    assert diagnostics.repeated_xml_lines == ['<has_text text="100.00"/>']
    assert diagnostics.missing_closing_tool is False


def test_diagnose_generated_xml_detects_repeated_cheetah_output_fragments() -> None:
    fragments = "\n".join(f"    #set $out_protocol_{index} = ''" for index in range(9))
    xml = f"""<tool id="adapter" name="Adapter" version="0.1.0">
    <command><![CDATA[
{fragments}
    ]]></command>
    <inputs/>
    <outputs/>
</tool>"""

    diagnostics = diagnose_generated_xml(xml)

    assert diagnostics.has_problems is True
    assert diagnostics.repeated_cheetah_fragments == 9
    assert diagnostics.repeated_cheetah_fragment_details == [
        {"fragment": "out_protocol", "count": 9}
    ]


def test_diagnose_generated_xml_detects_too_many_generated_tests() -> None:
    tests = "\n".join("<test><param name='input' value='reads.fq'/></test>" for _ in range(3))
    xml = f"""<tool id="map" name="Map" version="0.1.0">
    <command>minibwa map $input &gt; $out</command>
    <inputs><param name="input" type="data" format="fastq"/></inputs>
    <outputs><data name="out" format="bam"/></outputs>
    <tests>{tests}</tests>
</tool>"""

    diagnostics = diagnose_generated_xml(xml)

    assert diagnostics.has_problems is True
    assert diagnostics.test_count == 3
    assert diagnostics.too_many_tests is True
