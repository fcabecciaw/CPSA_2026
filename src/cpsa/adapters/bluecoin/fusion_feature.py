# fusion_feature.py
# Quaternion (compact) feature: firmware sends qx,qy,qz as int16 scaled by 10000.
# Reconstruct qw >= 0 as sqrt(1 - x^2 - y^2 - z^2) in the same raw domain.
#
# Created based on STM guide at https://github.com/STMicroelectronics/BlueSTSDK_Python?tab=readme-ov-file#how-to-add-a-new-feature
#
# Author: Francesco Urru
# GitHub: https://github.com/frarvo
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

import math

from blue_st_sdk.feature import ExtractedData, Feature, Sample
from blue_st_sdk.features.field import Field, FieldType
from blue_st_sdk.utils.blue_st_exceptions import (
    BlueSTInvalidDataException,
    BlueSTInvalidOperationException,
)
from blue_st_sdk.utils.number_conversion import LittleEndian


class FeatureMemsSensorFusionCompact(Feature):
    """
    Firmware sends (qx, qy, qz) as 3 × int16 scaled by 1/10000.
    qw is reconstructed as sqrt(1 - x² - y² - z²) with w >= 0 (sign folded by FW).
    """

    FEATURE_NAME = "Quaternion Compact"
    FEATURE_UNIT = ""
    FEATURE_DATA_NAME = ["QX", "QY", "QZ", "QW"]

    # Raw domain settings
    RAW_SCALE = 10000.0  # raw = float * 10000
    DATA_LENGTH_BYTES = 6  # 3 × int16 (2 bytes each)

    QX_INDEX = 0
    QY_INDEX = 1
    QZ_INDEX = 2
    QW_INDEX = 3

    FEATURE_QX_FIELD = Field(FEATURE_DATA_NAME[QX_INDEX], FEATURE_UNIT, FieldType.Float, +1.0, -1.0)
    FEATURE_QY_FIELD = Field(FEATURE_DATA_NAME[QY_INDEX], FEATURE_UNIT, FieldType.Float, +1.0, -1.0)
    FEATURE_QZ_FIELD = Field(FEATURE_DATA_NAME[QZ_INDEX], FEATURE_UNIT, FieldType.Float, +1.0, -1.0)
    FEATURE_QW_FIELD = Field(FEATURE_DATA_NAME[QW_INDEX], FEATURE_UNIT, FieldType.Float, +1.0, -1.0)

    def __init__(self, node):
        super().__init__(
            self.FEATURE_NAME,
            node,
            [
                self.FEATURE_QX_FIELD,
                self.FEATURE_QY_FIELD,
                self.FEATURE_QZ_FIELD,
                self.FEATURE_QW_FIELD,
            ],
        )

    def extract_data(self, timestamp, data, offset):
        """Extract qx,qy,qz (int16) and reconstruct |qw| in the same raw scaling."""
        if len(data) - offset < self.DATA_LENGTH_BYTES:
            raise BlueSTInvalidDataException(
                f"There are not {self.DATA_LENGTH_BYTES} bytes available to read."
            )

        # 3 × int16 at offsets 0,2,4
        qx_raw = LittleEndian.bytes_to_int16(data, offset + 0)
        qy_raw = LittleEndian.bytes_to_int16(data, offset + 2)
        qz_raw = LittleEndian.bytes_to_int16(data, offset + 4)

        # Reconstruct |qw| in raw (×10000) domain; clamp tiny negatives to 0
        t_raw = self.RAW_SCALE * self.RAW_SCALE - (
            qx_raw * qx_raw + qy_raw * qy_raw + qz_raw * qz_raw
        )
        if t_raw < 0:
            t_raw = 0
        qw_raw = int(math.sqrt(t_raw))

        sample = Sample(
            [float(qx_raw), float(qy_raw), float(qz_raw), float(qw_raw)],
            self.get_fields_description(),
            timestamp,
        )
        return ExtractedData(sample, self.DATA_LENGTH_BYTES)

    @classmethod
    def get_quaternion_qx(cls, sample):
        if sample and sample._data and sample._data[cls.QX_INDEX] is not None:
            return float(sample._data[cls.QX_INDEX])
        return float("nan")

    @classmethod
    def get_quaternion_qy(cls, sample):
        if sample and sample._data and sample._data[cls.QY_INDEX] is not None:
            return float(sample._data[cls.QY_INDEX])
        return float("nan")

    @classmethod
    def get_quaternion_qz(cls, sample):
        if sample and sample._data and sample._data[cls.QZ_INDEX] is not None:
            return float(sample._data[cls.QZ_INDEX])
        return float("nan")

    @classmethod
    def get_quaternion_qw(cls, sample):
        if sample and sample._data and sample._data[cls.QW_INDEX] is not None:
            return float(sample._data[cls.QW_INDEX])
        return float("nan")

    def read_quaternion(self):
        """Read the quaternion values (qx, qy, qz, |qw|) in raw ×10000 units."""
        try:
            self._read_data()
            sample = self._get_sample()
            return [
                self.get_quaternion_qx(sample),
                self.get_quaternion_qy(sample),
                self.get_quaternion_qz(sample),
                self.get_quaternion_qw(sample),
            ]
        except (BlueSTInvalidOperationException, BlueSTInvalidDataException) as e:
            raise e
