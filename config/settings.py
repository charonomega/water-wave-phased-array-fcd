"""轻量级 JSON 配置持久化管理。

提供类型安全、带默认值回退的键值存取接口，消除散布在 GUI 代码中的
try/except json.dump/json.load 样板代码。
"""

import json
import os


class AppConfig:
    """基于 JSON 文件的应用配置管理器。

    用法:
        cfg = AppConfig("my_config.json", defaults={"fps": 30.0})
        cfg.set("fps", 60.0)
        cfg.save()
        fps = cfg.get("fps", 30.0)
    """

    def __init__(self, filepath, defaults=None):
        self.filepath = filepath
        self.defaults = defaults or {}
        self._data = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def get(self, key, default=None):
        """获取配置值，优先使用已存储值，回退到构造函数默认值，最后回退到参数默认值。"""
        if key in self._data:
            return self._data[key]
        if key in self.defaults:
            return self.defaults[key]
        return default

    def set(self, key, value):
        """设置配置值（仅写内存，调用 save() 后落盘）。"""
        self._data[key] = value

    def update(self, mapping):
        """批量设置配置值。"""
        self._data.update(mapping)

    def save(self):
        """将所有配置持久化到 JSON 文件。"""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=4)
        except OSError:
            pass

    def get_all(self):
        """返回全部配置数据的浅拷贝。"""
        return dict(self._data)
