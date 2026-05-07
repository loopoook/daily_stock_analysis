# -*- coding: utf-8 -*-
"""
飞书 发送提醒服务

职责：
1. 通过 webhook 发送飞书群消息
2. 通过飞书应用 SDK 发送私聊消息（需配置 FEISHU_APP_ID + FEISHU_APP_SECRET + FEISHU_USER_ID）
"""
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

import requests

from src.config import Config
from src.formatters import (
    MIN_MAX_BYTES,
    PAGE_MARKER_SAFE_BYTES,
    chunk_content_by_max_bytes,
    format_feishu_markdown,
)

logger = logging.getLogger(__name__)

# 懒加载飞书 SDK，避免未安装时报错
_lark_sdk_available = None


def _check_lark_sdk():
    """检查 lark-oapi SDK 是否可用"""
    global _lark_sdk_available
    if _lark_sdk_available is None:
        try:
            import lark_oapi as lark  # noqa: F401
            _lark_sdk_available = True
        except ImportError:
            _lark_sdk_available = False
            logger.warning("lark-oapi SDK 未安装，飞书私聊推送不可用。请运行: pip install lark-oapi")
    return _lark_sdk_available


class FeishuSender:
    
    def __init__(self, config: Config):
        """
        初始化飞书配置

        支持两种推送模式：
        1. Webhook 模式：配置 FEISHU_WEBHOOK_URL，推送到群
        2. 应用 SDK 模式：配置 FEISHU_APP_ID + FEISHU_APP_SECRET + FEISHU_USER_ID，推送私聊

        当两种都配置时，优先使用 SDK 私聊推送。

        Args:
            config: 配置对象
        """
        # Webhook 模式配置
        self._feishu_url = getattr(config, 'feishu_webhook_url', None)
        self._feishu_secret = (getattr(config, 'feishu_webhook_secret', None) or '').strip()
        self._feishu_keyword = (getattr(config, 'feishu_webhook_keyword', None) or '').strip()
        self._feishu_max_bytes = getattr(config, 'feishu_max_bytes', 20000)
        self._webhook_verify_ssl = getattr(config, 'webhook_verify_ssl', True)

        # 应用 SDK 模式配置（私聊推送）
        self._feishu_app_id = getattr(config, 'feishu_app_id', None)
        self._feishu_app_secret = getattr(config, 'feishu_app_secret', None)
        self._feishu_user_id = getattr(config, 'feishu_user_id', None)

        # SDK 客户端（懒初始化）
        self._sdk_client = None

    @property
    def _use_sdk(self) -> bool:
        """判断是否使用 SDK 私聊模式"""
        return bool(
            self._feishu_app_id
            and self._feishu_app_secret
            and self._feishu_user_id
            and _check_lark_sdk()
        )

    def _get_sdk_client(self):
        """懒初始化飞书 SDK 客户端"""
        if self._sdk_client is not None:
            return self._sdk_client

        import lark_oapi as lark
        self._sdk_client = (
            lark.Client.builder()
            .app_id(self._feishu_app_id)
            .app_secret(self._feishu_app_secret)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )
        logger.info("飞书 SDK 客户端初始化成功（私聊推送模式）")
        return self._sdk_client

    def _get_keyword_prefix(self) -> str:
        """Return the keyword prefix required by Feishu webhook security settings."""
        if not self._feishu_keyword:
            return ""
        return f"{self._feishu_keyword}\n"

    def _apply_keyword_prefix(self, content: str) -> str:
        """Prepend the optional keyword so each webhook request passes keyword checks."""
        prefix = self._get_keyword_prefix()
        if not prefix:
            return content
        return f"{prefix}{content}" if content else self._feishu_keyword

    def _build_security_fields(self) -> Dict[str, str]:
        """Build optional signing fields required by Feishu custom robot security."""
        if not self._feishu_secret:
            return {}

        timestamp = str(int(time.time()))
        string_to_sign = f"{timestamp}\n{self._feishu_secret}"
        sign = base64.b64encode(
            hmac.new(
                string_to_sign.encode('utf-8'),
                digestmod=hashlib.sha256,
            ).digest()
        ).decode('utf-8')
        return {
            "timestamp": timestamp,
            "sign": sign,
        }
    
    def send_to_feishu(self, content: str) -> bool:
        """
        推送消息到飞书

        自动选择推送模式：
        - 配置了 APP_ID + APP_SECRET + USER_ID → SDK 私聊推送
        - 否则 → Webhook 群推送

        Args:
            content: 消息内容（Markdown 格式）

        Returns:
            是否发送成功
        """
        if self._use_sdk:
            return self._send_via_sdk(content)
        
        if not self._feishu_url:
            logger.warning("飞书 Webhook 未配置，跳过推送")
            return False
        
        return self._send_via_webhook(content)

    # ========== SDK 私聊推送 ==========

    def _send_via_sdk(self, content: str) -> bool:
        """
        通过飞书应用 SDK 发送私聊消息

        Args:
            content: Markdown 格式的消息内容

        Returns:
            是否发送成功
        """
        try:
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

            formatted_content = format_feishu_markdown(content)
            client = self._get_sdk_client()

            # 构建交互卡片（支持 Markdown 渲染）
            card_data = {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "股票智能分析报告"
                    }
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": formatted_content
                        }
                    }
                ]
            }

            # 检查是否需要分段发送
            content_bytes = len(formatted_content.encode('utf-8'))
            if content_bytes > self._feishu_max_bytes:
                return self._send_sdk_chunked(formatted_content)

            request = (
                CreateMessageRequest.builder()
                .receive_id_type("open_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(self._feishu_user_id)
                    .content(json.dumps(card_data))
                    .msg_type("interactive")
                    .build()
                )
                .build()
            )

            response = client.im.v1.message.create(request)

            if response.success():
                logger.info("飞书私聊消息发送成功 (user_id=%s)", self._feishu_user_id)
                return True
            else:
                logger.error(
                    "飞书私聊消息发送失败: code=%s, msg=%s, log_id=%s",
                    response.code, response.msg, response.get_log_id()
                )
                return False

        except Exception as e:
            logger.error("飞书 SDK 私聊推送异常: %s", e)
            return False

    def _send_sdk_chunked(self, content: str) -> bool:
        """
        SDK 模式分段发送长消息

        Args:
            content: 完整消息内容

        Returns:
            是否全部发送成功
        """
        try:
            chunks = chunk_content_by_max_bytes(content, self._feishu_max_bytes, add_page_marker=True)
        except ValueError as e:
            logger.error("飞书 SDK 消息分片失败: %s", e)
            return False

        success_count = 0
        total_chunks = len(chunks)
        logger.info("飞书 SDK 分批发送：共 %d 批", total_chunks)

        for i, chunk in enumerate(chunks):
            try:
                if self._send_single_sdk_message(chunk):
                    success_count += 1
                    logger.info("飞书 SDK 第 %d/%d 批发送成功", i + 1, total_chunks)
                else:
                    logger.error("飞书 SDK 第 %d/%d 批发送失败", i + 1, total_chunks)
            except Exception as e:
                logger.error("飞书 SDK 第 %d/%d 批发送异常: %s", i + 1, total_chunks, e)

            if i < total_chunks - 1:
                time.sleep(1)

        return success_count == total_chunks

    def _send_single_sdk_message(self, formatted_content: str) -> bool:
        """通过 SDK 发送单条消息"""
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        card_data = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": "股票智能分析报告"
                }
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": formatted_content
                    }
                }
            ]
        }

        client = self._get_sdk_client()
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(self._feishu_user_id)
                .content(json.dumps(card_data))
                .msg_type("interactive")
                .build()
            )
            .build()
        )

        response = client.im.v1.message.create(request)

        if response.success():
            return True
        else:
            logger.error(
                "飞书 SDK 发送失败: code=%s, msg=%s",
                response.code, response.msg
            )
            return False

    # ========== Webhook 群推送 ==========

    def _send_via_webhook(self, content: str) -> bool:
        """
        通过 Webhook 发送群消息

        Args:
            content: 消息内容

        Returns:
            是否发送成功
        """
        formatted_content = format_feishu_markdown(content)

        max_bytes = self._feishu_max_bytes
        keyword_overhead = len(self._get_keyword_prefix().encode('utf-8'))
        effective_max_bytes = max_bytes - keyword_overhead

        if effective_max_bytes <= 0:
            logger.error("飞书关键词过长，超过单条消息允许的最大字节数，无法发送")
            return False
        
        content_bytes = len(formatted_content.encode('utf-8')) + keyword_overhead
        if content_bytes > max_bytes:
            min_chunk_bytes = MIN_MAX_BYTES + PAGE_MARKER_SAFE_BYTES
            if effective_max_bytes < min_chunk_bytes:
                logger.error(
                    "飞书关键词过长，剩余分片预算(%s字节)不足以安全分页发送，至少需要 %s 字节",
                    effective_max_bytes,
                    min_chunk_bytes,
                )
                return False
            logger.info("飞书消息内容超长(%d字节/%d字符)，将分批发送", content_bytes, len(content))
            return self._send_webhook_chunked(formatted_content, effective_max_bytes)
        
        try:
            return self._send_webhook_message(formatted_content)
        except Exception as e:
            logger.error("发送飞书 Webhook 消息失败: %s", e)
            return False

    def _send_webhook_chunked(self, content: str, max_bytes: int) -> bool:
        """分批发送长消息到飞书 Webhook"""
        try:
            chunks = chunk_content_by_max_bytes(content, max_bytes, add_page_marker=True)
        except ValueError as e:
            logger.error("飞书 Webhook 消息分片失败: %s", e)
            return False

        total_chunks = len(chunks)
        success_count = 0

        logger.info("飞书 Webhook 分批发送：共 %d 批", total_chunks)

        for i, chunk in enumerate(chunks):
            try:
                if self._send_webhook_message(chunk):
                    success_count += 1
                    logger.info("飞书 Webhook 第 %d/%d 批发送成功", i + 1, total_chunks)
                else:
                    logger.error("飞书 Webhook 第 %d/%d 批发送失败", i + 1, total_chunks)
            except Exception as e:
                logger.error("飞书 Webhook 第 %d/%d 批发送异常: %s", i + 1, total_chunks, e)

            if i < total_chunks - 1:
                time.sleep(1)

        return success_count == total_chunks

    def _send_webhook_message(self, content: str) -> bool:
        """发送单条飞书 Webhook 消息（优先使用 Markdown 卡片）"""
        prepared_content = self._apply_keyword_prefix(content)
        security_fields = self._build_security_fields()

        def _post_payload(payload: Dict[str, Any]) -> bool:
            request_payload = dict(payload)
            request_payload.update(security_fields)
            logger.debug("飞书请求 URL: %s", self._feishu_url)
            logger.debug("飞书请求 payload 长度: %d 字符", len(prepared_content))

            response = requests.post(
                self._feishu_url,
                json=request_payload,
                timeout=30,
                verify=self._webhook_verify_ssl
            )

            logger.debug("飞书响应状态码: %d", response.status_code)
            logger.debug("飞书响应内容: %s", response.text)

            if response.status_code == 200:
                result = response.json()
                code = result.get('code') if 'code' in result else result.get('StatusCode')
                if code == 0:
                    logger.info("飞书 Webhook 消息发送成功")
                    return True
                else:
                    error_msg = result.get('msg') or result.get('StatusMessage', '未知错误')
                    error_code = result.get('code') or result.get('StatusCode', 'N/A')
                    logger.error("飞书返回错误 [code=%s]: %s", error_code, error_msg)
                    logger.error("完整响应: %s", result)
                    return False
            else:
                logger.error("飞书请求失败: HTTP %d", response.status_code)
                logger.error("响应内容: %s", response.text)
                return False

        # 1) 优先使用交互卡片（支持 Markdown 渲染）
        card_payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "股票智能分析报告"
                    }
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": prepared_content
                        }
                    }
                ]
            }
        }

        if _post_payload(card_payload):
            return True

        # 2) 回退为普通文本消息
        text_payload = {
            "msg_type": "text",
            "content": {
                "text": prepared_content
            }
        }

        return _post_payload(text_payload)
