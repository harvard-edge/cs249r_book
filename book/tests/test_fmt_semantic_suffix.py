from book.cli.checks.fmt_semantic_suffix import audit


def _write(tmp_path, body):
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        "\n```{python}\nfrom mlsysim.fmt import fmt, fmt_int, fmt_usd\n"
        + body
        + "\n```\n",
        encoding="utf-8",
    )
    return chapter


def test_flags_percent_suffix(tmp_path):
    chapter = _write(
        tmp_path,
        "x_str = fmt(acc * 100, precision=0, commas=False, suffix=' percent')\n"
        "y_str = fmt(rate, precision=1, commas=False, suffix='%')\n",
    )
    codes = [v.code for v in audit([chapter])]
    assert codes.count("percent_in_suffix") == 2


def test_flags_multiplier_suffix(tmp_path):
    chapter = _write(
        tmp_path,
        "a_str = fmt(speedup, precision=1, commas=False, suffix='×')\n"
        "b_str = fmt(ratio, precision=0, commas=False, suffix='x')\n",
    )
    codes = [v.code for v in audit([chapter])]
    assert codes.count("multiplier_in_suffix") == 2


def test_flags_percentage_points_suffix(tmp_path):
    chapter = _write(
        tmp_path,
        "g_str = fmt(gap, precision=0, commas=False, suffix=' percentage points')\n"
        "h_str = fmt(gap, precision=0, commas=False, suffix=' pp')\n",
    )
    codes = [v.code for v in audit([chapter])]
    assert codes.count("pp_in_suffix") == 2


def test_flags_scale_glyph_on_fmt(tmp_path):
    chapter = _write(
        tmp_path,
        "q_str = fmt(queries / MILLION, precision=0, commas=False, suffix='M')\n",
    )
    codes = [v.code for v in audit([chapter])]
    assert codes.count("scale_glyph_in_suffix") == 1


def test_flags_scale_word_on_fmt(tmp_path):
    chapter = _write(
        tmp_path,
        "q_str = fmt(queries / MILLION, precision=0, commas=False, suffix=' million')\n",
    )
    codes = [v.code for v in audit([chapter])]
    assert codes.count("scale_word_in_suffix") == 1


def test_flags_service_rate_suffix(tmp_path):
    chapter = _write(
        tmp_path,
        "r_str = fmt(rate, precision=0, commas=False, suffix=' tokens/s')\n"
        "f_str = fmt(frames, precision=0, commas=False, suffix=' FPS')\n",
    )
    codes = [v.code for v in audit([chapter])]
    assert codes.count("rate_in_suffix") == 2


def test_does_not_flag_physical_unit_suffix(tmp_path):
    chapter = _write(
        tmp_path,
        "m_str = fmt(mem_gb, precision=1, commas=False, suffix=' GB')\n"
        "t_str = fmt(lat_ms, precision=0, commas=False, suffix=' ms')\n"
        "r_str = fmt(bw, precision=1, commas=False, suffix=' GB/s')\n",
    )
    assert audit([chapter]) == []


def test_does_not_flag_fmt_usd_scale_suffix(tmp_path):
    # Currency scale belongs on fmt_usd; this checker only inspects fmt/fmt_int.
    chapter = _write(
        tmp_path,
        "c_str = fmt_usd(cost_m, precision=1, suffix='M')\n",
    )
    assert audit([chapter]) == []
