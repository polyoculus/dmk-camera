"""Direct access to the TIS vendor UVC extension unit on the DMK 33UX273.

The standard UVC controls don't expose auto-gain (it stays on and quietly
fights manual settings) and only give exposure in 100 us steps. The vendor
extension unit has both, so we talk to it with UVCIOC_CTRL_QUERY - no driver
or root needed. Selectors are from tiscamera's data/uvc-extensions/usb33.json.
"""

import ctypes
import fcntl
import os

UNIT = 3  # XU id from the VC descriptor, GUID 0aba49de-5c0b-49d5-8f71-0be40f94a67a

AUTO_SHUTTER = 0x01
GAIN_AUTO = 0x02
EXPOSURE_US = 0x1B
GAIN_CDB = 0x18  # dB/100

SET_CUR, GET_CUR, GET_MIN, GET_MAX, GET_LEN = 0x01, 0x81, 0x82, 0x83, 0x85


class _Query(ctypes.Structure):
    _fields_ = [("unit", ctypes.c_uint8), ("selector", ctypes.c_uint8),
                ("query", ctypes.c_uint8), ("size", ctypes.c_uint16),
                ("data", ctypes.POINTER(ctypes.c_uint8))]


# _IOWR('u', 0x21, struct uvc_xu_control_query)
UVCIOC_CTRL_QUERY = (3 << 30) | (ctypes.sizeof(_Query) << 16) | (ord("u") << 8) | 0x21


class XU:
    def __init__(self, dev):
        self.fd = os.open(dev, os.O_RDWR)
        self._len = {}

    def _q(self, sel, query, size, value=0):
        buf = (ctypes.c_uint8 * size)(*value.to_bytes(size, "little"))
        fcntl.ioctl(self.fd, UVCIOC_CTRL_QUERY, _Query(UNIT, sel, query, size, buf))
        return int.from_bytes(bytes(buf), "little")

    def size(self, sel):
        if sel not in self._len:
            self._len[sel] = self._q(sel, GET_LEN, 2)
        return self._len[sel]

    def get(self, sel, query=GET_CUR):
        return self._q(sel, query, self.size(sel))

    def set(self, sel, value):
        self._q(sel, SET_CUR, self.size(sel), int(value))

    def range(self, sel):
        return self.get(sel, GET_MIN), self.get(sel, GET_MAX)

    def close(self):
        os.close(self.fd)
