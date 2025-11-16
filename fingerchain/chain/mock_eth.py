"""用于本地测试的以太坊交互 Mock，实现和真实链同样的接口。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MockTransaction:
    tx_hash: str
    timestamp: float
    payload: Dict[str, Any]


@dataclass
class MockEthChainAPI:
    owner_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    user_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    media_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    share_log: List[Dict[str, Any]] = field(default_factory=list)

    def _fake_tx(self, payload: Dict[str, Any]) -> MockTransaction:
        """构造一个假的交易对象，包含时间戳和 payload。"""
        return MockTransaction(tx_hash=uuid.uuid4().hex, timestamp=time.perf_counter(), payload=payload)

    def register_owner(self, info: Dict[str, Any]) -> MockTransaction:
        """记录 owner 数据，返回假交易。"""
        self.owner_registry[info["address"]] = info
        return self._fake_tx(info)

    def register_user(self, info: Dict[str, Any]) -> MockTransaction:
        """记录 user 数据。"""
        self.user_registry[info["address"]] = info
        return self._fake_tx(info)

    def upload_media(self, info: Dict[str, Any]) -> MockTransaction:
        """记录媒体元数据及其 IPFS hash。"""
        self.media_registry[info["media_key"]] = info
        return self._fake_tx(info)

    def share_media(self, info: Dict[str, Any]) -> MockTransaction:
        """追加一条分享记录以模拟 MediaShare 交易。"""
        self.share_log.append(info)
        return self._fake_tx(info)
