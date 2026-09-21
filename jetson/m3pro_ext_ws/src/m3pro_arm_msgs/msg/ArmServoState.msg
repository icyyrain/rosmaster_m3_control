# Raw bus-servo feedback produced by the STM32 micro-ROS firmware.
# Bit i of valid_mask corresponds to position_raw[i].
uint32 seq
uint32 sample_time_ms
int16[6] position_raw
uint8 valid_mask
