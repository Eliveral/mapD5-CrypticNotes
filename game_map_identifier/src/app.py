from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import ConfigError, load_app_config_for_first_run, load_project_config
from .first_run_setup import ensure_first_run_feature_area
from .gui import GameMapIdentifierGUI
from .map_database import MapDatabase, MapDatabaseError, create_orb_from_config
from .preprocess import preprocess_config_from_dict
from .runtime_diagnostics import diagnostic_event, install_tk_exception_handler, mark_clean_exit
from .sample_database import ScreenshotSampleDatabase
from .utils import setup_logging
from .windows import enable_physical_pixel_coordinates


def should_load_runtime_sample_database(app_config: dict) -> bool:
    """Historical screenshot matching is optional and unnecessary in CNN-only mode."""
    return bool(app_config.get("enable_screenshot_sample_match", False)) and not bool(
        app_config.get("cnn_only_mode", False)
    )


def main(app_config_path: Path | None = None) -> int:
    enable_physical_pixel_coordinates()
    setup_logging()
    diagnostic_event("application_initializing", config_path=str(app_config_path) if app_config_path else None)
    try:
        first_run_config_path, first_run_app_config = load_app_config_for_first_run(app_config_path)
        if not ensure_first_run_feature_area(first_run_config_path, first_run_app_config):
            return 1
        project_config = load_project_config(app_config_path=app_config_path)
        preprocess_config = preprocess_config_from_dict(project_config.app.get("preprocess"))
        orb = create_orb_from_config(project_config.app)
        db = MapDatabase(project_config.maps, preprocess_config, orb).load(strict=False)
        sample_db = ScreenshotSampleDatabase(project_config.root_dir, project_config.maps, preprocess_config)
        if should_load_runtime_sample_database(project_config.app):
            sample_db.load()
    except (ConfigError, MapDatabaseError) as exc:
        diagnostic_event("application_initialization_failed", exception=repr(exc))
        logging.error("%s", exc)
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2

    diagnostic_event("configuration_loaded", root_dir=str(project_config.root_dir))
    app = GameMapIdentifierGUI(project_config, preprocess_config, db, sample_db)
    install_tk_exception_handler(app)
    diagnostic_event("mainloop_enter")
    app.mainloop()
    diagnostic_event("mainloop_returned")
    mark_clean_exit("mainloop_returned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
