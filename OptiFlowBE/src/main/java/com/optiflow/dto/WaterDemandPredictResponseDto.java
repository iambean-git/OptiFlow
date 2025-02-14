package com.optiflow.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Data
@Getter @Setter
public class WaterDemandPredictResponseDto {
	
	@JsonProperty("prediction") // JSON 키 "prediction" 과 매핑
	private List<WaterDemandPredictionItemDto> prediction;
	
	@JsonProperty("optiflow")
	private List<WaterDemandPredictionItemDto> optiflow;
    
	public WaterDemandPredictResponseDto() {
		this.prediction = java.util.Collections.emptyList();
		this.optiflow = java.util.Collections.emptyList();
	}
	
}
