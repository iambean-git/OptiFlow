package com.optiflow.dto;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class WaterDemandPredictionItemDto {

    @JsonProperty("time") // JSON 키 "time" 과 매핑
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime time;

    @JsonProperty("value")
    private Double value;
    
    @JsonProperty("height")
    private Double height;
    
    public WaterDemandPredictionItemDto() {
    }
}
