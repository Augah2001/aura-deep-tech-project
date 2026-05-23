from __future__ import annotations

import time
from typing import Any


class ArduinoSerialBridge:
    def __init__(self) -> None:
        self.port = "COM16"
        self.baud_rate = 115200
        self.connected = False
        self.last_error: str | None = None
        self.last_ack: str | None = None
        self.last_sync: str | None = None
        self._serial: Any = None

    def connect(self, port: str | None = None, baud_rate: int | None = None) -> dict[str, Any]:
        self.port = port or self.port
        self.baud_rate = int(baud_rate or self.baud_rate)
        try:
            import serial  # type: ignore
        except Exception as exc:
            self.connected = False
            self.last_error = f"pyserial unavailable: {exc}"
            return self.status()

        try:
            self.disconnect()
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2.0)
            self.connected = True
            self.last_error = None
            self.last_ack = None
        except Exception as exc:
            self._serial = None
            self.connected = False
            self.last_error = f"{type(exc).__name__}: {exc}"
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self.connected = False
        self.last_error = None
        return self.status()

    def sync(self, command: str) -> dict[str, Any]:
        if not self.connected or self._serial is None:
            self.last_error = "serial bridge is not connected"
            return self.status()
        try:
            payload = f"{command}\n".encode("utf-8")
            self._serial.write(payload)
            self._serial.flush()
            ack = self._serial.readline().decode("utf-8", errors="replace").strip()
            self.last_ack = ack or "sent"
            self.last_sync = time.strftime("%Y-%m-%d %H:%M:%S")
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.connected = False
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "port": self.port,
            "baud_rate": self.baud_rate,
            "last_error": self.last_error,
            "last_ack": self.last_ack,
            "last_sync": self.last_sync,
        }


bridge = ArduinoSerialBridge()
