"""Logging central da V9.2."""
from __future__ import annotations
import json, logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({'level':record.levelname,'logger':record.name,'message':record.getMessage(),'time':self.formatTime(record)},ensure_ascii=False)

def configure_logging(*,verbose=False,quiet=False,json_format=False,log_dir=None):
    level=logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    root=logging.getLogger(); root.setLevel(level)
    if not root.handlers:
        stream=logging.StreamHandler(); stream.setFormatter(JsonFormatter() if json_format else logging.Formatter('%(levelname)s %(name)s: %(message)s')); root.addHandler(stream)
    out=Path(log_dir or Path(__file__).resolve().parent.parent/'results'/'logs'); out.mkdir(parents=True,exist_ok=True)
    if not any(isinstance(h,RotatingFileHandler) for h in root.handlers):
        fh=RotatingFileHandler(out/'biuri.log',maxBytes=2_000_000,backupCount=3,encoding='utf-8'); fh.setFormatter(JsonFormatter() if json_format else logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')); root.addHandler(fh)
    return root
