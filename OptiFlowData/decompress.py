import base64
import gzip
import json
import numpy as np


# 데이터 압축 및 압축 해제 함수
def decompress(data):
    """데이터를 Base64 및 gzip 방식으로 디코딩하여 반환.
        TB_MOTOR 테이블 DATA_ARRAY 시용
        
        -- TB_MOTOR definition

        CREATE TABLE `TB_MOTOR` (
          `MOTOR_ID` varchar(30) NOT NULL COMMENT '모터ID',
          `EQUIPMENT_ID` varchar(30) NOT NULL COMMENT '장비 KEY',
          `CENTER_ID` varchar(20) NOT NULL COMMENT '센터 ID',
          `CHANNEL_ID` varchar(20) NOT NULL COMMENT '채널 KEY',
          `ACQ_DATE` timestamp NOT NULL DEFAULT current_timestamp() COMMENT '계측 시간',
          `DATA_ARRAY` mediumtext DEFAULT NULL COMMENT '진동 데이터',
          `PROC_STAT` int(11) DEFAULT 0 COMMENT '알고리즘 동작 여부',
          PRIMARY KEY (`MOTOR_ID`,`EQUIPMENT_ID`,`CENTER_ID`,`CHANNEL_ID`,`ACQ_DATE`),
          KEY `IDX_TB_MOTER_ACQ_DATE` (`ACQ_DATE`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLAT
    
    """
    try:
        byte_data = gzip.decompress(base64.standard_b64decode(data))
        temp = byte_data.decode("utf-16")
        temp_jarr = json.loads(temp)
        return np.array(list(map(lambda x: float(x), temp_jarr)))
    except Exception as e:
        logger.error(f"Data decompression error: {e}")
        raise