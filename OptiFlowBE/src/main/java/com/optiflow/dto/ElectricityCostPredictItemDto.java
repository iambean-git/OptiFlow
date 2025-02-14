package com.optiflow.dto;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class ElectricityCostPredictItemDto {
	
    @JsonProperty("time") // JSON 키 "time" 과 매핑
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime time; // LocalDateTime 타입으로 변경

    @JsonProperty("value")
    private Integer value;
    
}
