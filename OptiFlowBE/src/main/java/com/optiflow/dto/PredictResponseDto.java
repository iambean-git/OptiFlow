package com.optiflow.dto;

import java.time.LocalDateTime;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Data
@Getter @Setter
public class PredictResponseDto {
	
	@JsonProperty("prediction") // JSON 키 "prediction" 과 매핑
	private List<PredictionItemDto> prediction;
	
	@JsonProperty("optiflow")
	private List<PredictionItemDto> optiflow;
	
	@JsonProperty("time") // JSON 키 "time" 과 매핑
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime time; // LocalDateTime 타입으로 변경

    @JsonProperty("value") // JSON 키 "value" 와 매핑
    private Double value;
    
	public PredictResponseDto() {
		this.prediction = java.util.Collections.emptyList();
		this.optiflow = java.util.Collections.emptyList();
	}
	
}
