# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import logging
import sys

from colorama import Fore, Style, init

init()


class ColoredFormatter(logging.Formatter):
    """Custom formatter with error handling and colored output"""

    format_str = "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

    FORMATS = {
        logging.DEBUG: Fore.CYAN + format_str + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + format_str + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + format_str + Style.RESET_ALL,
        logging.ERROR: Fore.RED + format_str + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + format_str + Style.RESET_ALL,
    }

    def format(self, record):
        try:
            formatter = logging.Formatter(self.FORMATS[record.levelno])
            return formatter.format(record)
        except Exception as e:
            try:
                return (
                    f"{Fore.RED}Log format error: {e} | "
                    f"Original message: {repr(record.msg)} | "
                    f"Args: {repr(record.args)}{Style.RESET_ALL}"
                )
            except Exception:
                return f"{Fore.RED}Critical log formatting failure{Style.RESET_ALL}"


class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        self.logger = logging.getLogger("G2Robot")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredFormatter())
            self.logger.addHandler(console_handler)

        self._wrap_logging_methods()

    def _wrap_logging_methods(self):
        methods = ["debug", "info", "warning", "error", "critical", "exception"]
        for method_name in methods:
            original = getattr(self.logger, method_name)
            setattr(self, method_name, self._create_safe_method(original))

    def _create_safe_method(self, method):
        def safe_method(msg, *args, **kwargs):
            try:
                method(msg, *args, **kwargs)
            except Exception as e:
                try:
                    self.logger.error(f"Logging error: {e} [Message: {repr(msg)}, Args: {repr(args)}]")
                except Exception:
                    self.logger.error("Critical failure in error handling")

        return safe_method


logger = Logger()
