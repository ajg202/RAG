from docqa.config import PROJECT_ROOT, settings


def test_project_root_contains_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_paths_are_absolute_and_inside_project():
    assert settings.chroma_dir.is_absolute()
    assert settings.chroma_dir.is_relative_to(PROJECT_ROOT)


def test_numeric_settings_are_sane():
    assert settings.chunk_max_chars > settings.chunk_overlap >= 0
    assert settings.top_k > 0