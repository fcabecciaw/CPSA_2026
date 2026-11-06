"""Process-wide lock serializing BLE reconnection attempts.

BlueZ handles concurrent connection attempts poorly, so every BLE device thread
(BlueCoin, MetaMotion, speaker, LED) reconnects while holding this lock.
"""

import threading

ble_reconnection_lock = threading.RLock()
