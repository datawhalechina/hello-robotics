# This software contains source code provided by NVIDIA Corporation.
# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Isaac Sim application launcher wrapper."""

import argparse
import contextlib
import os
import signal
from typing import Any, Literal

from g2_robot.utils.logger import Logger

logger = Logger()

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim

from isaacsim import SimulationApp


class AppLauncher:
    """Launches Isaac Sim SimulationApp with the given configuration."""

    def __init__(self, app_config):
        self._headless: bool
        self._render_mode = "RealTimePathTracing"

        self._config_resolution(app_config)
        self._create_app()

        signal.signal(signal.SIGINT, self._interrupt_signal_handle_callback)
        signal.signal(signal.SIGTERM, self._abort_signal_handle_callback)
        signal.signal(signal.SIGABRT, self._abort_signal_handle_callback)
        signal.signal(signal.SIGSEGV, self._abort_signal_handle_callback)

    @property
    def app(self) -> SimulationApp:
        if self._app is not None:
            return self._app
        else:
            raise RuntimeError("The `AppLauncher.app` member cannot be retrieved until the class is initialized.")

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
        parser_help = None
        if len(parser._actions) > 0 and isinstance(parser._actions[0], argparse._HelpAction):
            parser_help = parser._actions[0]
            parser._option_string_actions.pop("-h")
            parser._option_string_actions.pop("--help")

        known, _ = parser.parse_known_args()
        config = vars(known)
        if len(config) == 0:
            logger.warning(
                "[WARN][AppLauncher]: There are no arguments attached to the ArgumentParser object."
                " If you have your own arguments, please load your own arguments before calling the"
                " `AppLauncher.add_app_launcher_args` method."
            )
        else:
            AppLauncher._check_argparser_config_params(config)

        arg_group = parser.add_argument_group(
            "app_launcher arguments",
            description="Arguments for the AppLauncher.",
        )
        arg_group.add_argument("--headless", action="store_true", default=False)
        arg_group.add_argument("--livestream", type=int, default=-1, choices={0, 1, 2})
        arg_group.add_argument("--enable_cameras", action="store_true", default=False)
        arg_group.add_argument("--device", type=str, default="cuda:0")
        arg_group.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)
        arg_group.add_argument("--verbose", action="store_true")
        arg_group.add_argument("--info", action="store_true")
        arg_group.add_argument("--experience", type=str, default="")
        arg_group.add_argument("--kit_args", type=str, default="")

        if parser_help is not None:
            parser._option_string_actions["-h"] = parser_help
            parser._option_string_actions["--help"] = parser_help

    @staticmethod
    def _check_argparser_config_params(config: dict) -> None:
        applauncher_keys = set(AppLauncher._APPLAUNCHER_CFG_INFO.keys())
        for key, value in config.items():
            if key in applauncher_keys:
                raise ValueError(
                    f"The passed ArgParser object already has the field '{key}'."
                    " This field will be added by `AppLauncher.add_app_launcher_args()`."
                )
        simulationapp_keys = set(AppLauncher._SIM_APP_CFG_TYPES.keys())
        for key, value in config.items():
            if key in simulationapp_keys:
                given_type = type(value)
                expected_types = AppLauncher._SIM_APP_CFG_TYPES[key]
                if type(value) not in set(expected_types):
                    raise ValueError(
                        f"Invalid value type for the argument '{key}': {given_type}."
                        f" Expected one of {expected_types}."
                    )

    def _config_resolution(self, app_config):
        self._headless = app_config.headless
        self._render_mode = app_config.render_mode

    def _create_app(self):
        self._app = SimulationApp(
            {
                "headless": self._headless,
                "disable_viewport_updates": self._headless,
                "renderer": self._render_mode,
                "limit_cpu_threads": 16,
                "extra_args": [],
            }
        )

    _APPLAUNCHER_CFG_INFO: dict[str, tuple[list[type], Any]] = {
        "headless": ([bool], False),
        "livestream": ([int], -1),
        "enable_cameras": ([bool], False),
        "device": ([str], "cuda:0"),
        "experience": ([str], ""),
    }

    _SIM_APP_CFG_TYPES: dict[str, list[type]] = {
        "headless": [bool],
        "hide_ui": [bool, type(None)],
        "active_gpu": [int, type(None)],
        "physics_gpu": [int],
        "multi_gpu": [bool],
        "sync_loads": [bool],
        "width": [int],
        "height": [int],
        "window_width": [int],
        "window_height": [int],
        "display_options": [int],
        "subdiv_refinement_level": [int],
        "renderer": [str],
        "anti_aliasing": [int],
        "samples_per_pixel_per_frame": [int],
        "denoiser": [bool],
        "max_bounces": [int],
        "max_specular_transmission_bounces": [int],
        "max_volume_bounces": [int],
        "open_usd": [str, type(None)],
        "livesync_usd": [str, type(None)],
        "fast_shutdown": [bool],
        "experience": [str],
    }

    def _interrupt_signal_handle_callback(self, signal, frame):
        self._app.close()
        raise KeyboardInterrupt

    def _abort_signal_handle_callback(self, signal, frame):
        self._app.close()
