from src.config import (
    PROJECT_ROOT,
    PDF_DIR,
    VECTOR_DB_DIR,
    APP_NAME,
    validate_config,
)


def test_project_root_exists():
    """Check that the project root exists."""
    assert PROJECT_ROOT.exists()


def test_pdf_directory_exists():
    """Check that the PDF directory exists."""
    assert PDF_DIR.exists()


def test_vector_db_directory_exists():
    """Check that the vector database directory exists."""
    assert VECTOR_DB_DIR.exists()


def test_app_name_exists():
    """Check that the application has a name."""
    assert APP_NAME


def test_config_is_valid():
    """Check that the configuration passes validation."""
    assert validate_config() is True