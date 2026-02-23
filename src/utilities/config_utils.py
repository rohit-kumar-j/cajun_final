"""<src/utilities/config_utils.py>
Shared config-reading utility for the AI framework.

All config reads that happen at initialisation time (not in hot loops)
must go through cfg() so that:
  1. Missing keys are logged with a clear WARNING including caller file + line
  2. The fallback value used is explicitly documented
  3. config.warnings.<section>.found_<key> is set for visualiser inspection
  4. Behaviour is consistent across all classes

Usage:
    from src.utilities.config_utils import cfg, flush_warnings, REQUIRED

    # In __init__ only, never in step() or other hot-path methods:
    foot_friction = cfg(config, "foot_friction", 1.0, "JumpEnv", warn_flags)
    urdf_path     = cfg(config, "urdf_path",     REQUIRED, "JumpEnv", warn_flags)

    flush_warnings(config, "JumpEnv", warn_flags)

Hot-path methods must NOT call cfg(). Use self._ attributes cached in __init__.
"""
import inspect
import os

from loguru import logger
import ml_collections


# Sentinel — pass as fallback to indicate the key is required (will raise).
# This allows None to be a valid fallback (optional feature disabled by None).
REQUIRED = object()


def cfg(config, key: str, fallback, caller: str, warn_flags: dict):
    """Read one config value with logging, caller location, and warning tracking.

    Args:
        config:     ml_collections.ConfigDict to read from
        key:        attribute name to look up
        fallback:   REQUIRED  → raise AttributeError if missing
                    None      → return None silently (optional feature)
                    <value>   → return <value> with a WARNING if missing
        caller:     display name, e.g. "JumpEnv" or "config.robot"
        warn_flags: accumulates found_<key> booleans for flush_warnings()

    Returns:
        Config value if found, else fallback.

    Raises:
        AttributeError if key is missing and fallback is REQUIRED.
    """
    try:
        val = getattr(config, key)
        warn_flags[f"found_{key}"] = True
        return val
    except AttributeError:
        warn_flags[f"found_{key}"] = False

        # Locate the call site (skip this frame and any internal inspect frames)
        frame     = inspect.currentframe().f_back
        filename  = os.path.basename(frame.f_code.co_filename)
        lineno    = frame.f_lineno
        location  = f"{filename}:{lineno}"

        if fallback is REQUIRED:
            logger.error(
                f"[{caller}] ({location}) config.{key} NOT FOUND — "
                f"this key is required. Add {key} to the config."
            )
            raise AttributeError(
                f"[{caller}] required config key '{key}' is missing"
            )

        if fallback is None:
            logger.warning(
                f"[{caller}] ({location}) [{key}] NOT SET (optional, defaulting to None)"
            )
        else:
            logger.warning(
                f"[{caller}] ({location}) config.{key} NOT FOUND — "
                f"using fallback: {fallback!r}. "
                f"Add {key} to the config to suppress this warning."
            )
        return fallback


def flush_warnings(config, section_name: str, warn_flags: dict) -> None:
    """Write found_* flags into config.warnings for visualiser inspection.

    Any found_* = False means that key was auto-filled with a fallback.

    Args:
        config:       the config object to write warnings into
        section_name: human-readable label for log messages
        warn_flags:   dict of {found_<key>: bool} from cfg() calls
    """
    if not warn_flags:
        return

    missing = [k for k, v in warn_flags.items() if not v]

    try:
        with config.unlocked():
            if not hasattr(config, "warnings"):
                config.warnings = ml_collections.ConfigDict()
            with config.warnings.unlocked():
                for k, v in warn_flags.items():
                    config.warnings[k] = v
    except Exception as exc:
        logger.warning(
            f"[config_utils] Could not write warnings to "
            f"config.{section_name}.warnings: {exc}"
        )
        return

    if missing:
        logger.warning(
            f"[{section_name}] Auto-filled {len(missing)} missing config "
            f"key(s): {missing}"
        )
    else:
        logger.success(f"[{section_name}] All config values explicitly set")

